#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
# This code is basically a supplementary material of the book
# "深層学習による自然言語処理 (機械学習プロフェッショナルシリーズ)
# ISBN-10: 4061529242".
# This is the reason why the comments in this code were written in Japanese,
# and written in single file.


# seq2seq-attn(またはopenNMT)をベースにして作成
# 動作確認
# * python2.7 と 3.5, 3.6
# * chainer v1.24 と v2.0
# * bottleneckのインストールを推奨
#   https://pypi.python.org/pypi/Bottleneck
#
# * サンプルの利用法
# サンプルデータはWMT16のページから入手
# http://www.statmt.org/wmt16/translation-task.html
# 学習 (GPUが無い環境では以下でGPU=-1として実行)
SLAN=de; TLAN=en; GPU=0;  EP=13 ;  \
MODEL=filename_of_sample_model.model ; \
python -u ./LSTMEncDecAttn.py -V2 \
   -T                      train \
   --gpu-enc               ${GPU} \
   --gpu-dec               ${GPU} \
   --enc-vocab-file        sample_data/newstest2012-4p.${SLAN}.vocab_t3_tab \
   --dec-vocab-file        sample_data/newstest2012-4p.${TLAN}.vocab_t3_tab \
   --enc-data-file         sample_data/newstest2012-4p.${SLAN} \
   --dec-data-file         sample_data/newstest2012-4p.${TLAN} \
   --enc-devel-data-file   sample_data/newstest2015.h100.${SLAN} \
   --dec-devel-data-file   sample_data/newstest2015.h100.${TLAN} \
   -D                          512 \
   -H                          512 \
   -N                          2 \
   --optimizer                 SGD \
   --lrate                     1.0 \
   --batch-size                32 \
   --out-each                  0 \
   --epoch                     ${EP} \
   --eval-accuracy             0 \
   --dropout-rate              0.3 \
   --attention-mode            1 \
   --gradient-clipping         5 \
   --initializer-scale         0.1 \
   --initializer-type          uniform \
   --merge-encoder-fwbw        0 \
   --use-encoder-bos-eos       0 \
   --use-decoder-inputfeed     1 \
   -O                          ${MODEL} \


# 評価
SLAN=de; GPU=0;  EP=13 ; BEAM=5 ;  \
MODEL=filename_of_sample_model.model ; \
python -u ./LSTMEncDecAttn.py \
   -T                  test \
   --gpu-enc           ${GPU} \
   --gpu-dec           ${GPU} \
   --enc-data-file     sample_data/newstest2015.h101-200.${SLAN} \
   --init-model        ${MODEL}.epoch${EP} \
   --setting           ${MODEL}.setting    \
   --beam-size         ${BEAM} \
   --max-length        150 \
   > ${MODEL}.epoch${EP}.decode_MAX${MAXLEN}_BEAM${BEAM}.txt
'''


import sys
import collections
import six.moves.cPickle as pickle
import copy
import numpy as np
import bottleneck as bn
import argparse
import time
import random
import math
import six
import io
import codecs

import chainer
import chainer.functions as chaFunc
import chainer.optimizers as chaOpt
import chainer.links as chaLink
import chainer.serializers as chaSerial
from chainer import cuda

# chainer v4対応


def _sum_sqnorm(arr):
    sq_sum = collections.defaultdict(float)
    for x in arr:
        with cuda.get_device_from_array(x) as dev:
            x = x.ravel()
            s = x.dot(x)
            sq_sum[int(dev)] += s
    return sum([float(i) for i in six.itervalues(sq_sum)])


# gradientのnormなどを効率的に取得するための処理
# logの出力で使わないなら，本来なくてもいい部分
class Chainer_GradientClipping_rmk_v1(chainer.optimizer.GradientClipping):
    name = 'GradientClipping'

    def __init__(self, threshold):
        self.threshold = threshold
        self.rate = 1
        self.norm = 1
        self.norm_orig = 1

    def __call__(self, opt):
        self.norm_orig = np.sqrt(_sum_sqnorm(
            [p.grad for p in opt.target.params()]))
        self.norm = self.norm_orig
        self.rate = self.threshold / self.norm_orig
        if self.rate < 1:
            for param in opt.target.params():
                grad = param.grad
                with cuda.get_device(grad):
                    grad *= self.rate
            self.norm = self.threshold


# LSTMの層の数を変数で決定したいので，層数が可変なことをこのクラスで吸収する
# ここでは主にdecoder用のLSTMを構築するために利用
class NLayerLSTM(chainer.ChainList):

    def __init__(self, n_layers=2, eDim=512, hDim=512, name=""):
        layers = [0] * n_layers  # 層分の領域を確保
        for z in six.moves.range(n_layers):
            if z == 0:  # 第一層の次元数は eDim
                tDim = eDim
            else:  # 第二層以上は前の層の出力次元が入力次元となるのでhDim
                tDim = hDim
            layers[z] = chaLink.LSTM(tDim, hDim)
            # logに出力する際にわかりやすくするための名前付け
            layers[z].lateral.W.name = name + "_L%d_la_W" % (z + 1)
            layers[z].upward.W.name = name + "_L%d_up_W" % (z + 1)
            layers[z].upward.b.name = name + "_L%d_up_b" % (z + 1)

        super(NLayerLSTM, self).__init__(*layers)

    # 全ての層に対して一回だけLSTMを回す
    def processOneStepForward(self, input_states, args, dropout_rate):
        hout = None
        for c, layer in enumerate(self):
            if c > 0:  # 一層目(embedding)の入力に対してはdropoutしない
                hin = chaFunc.dropout(hout, ratio=dropout_rate)
            else:  # 二層目以降の入力はdropoutする
                hin = input_states
            hout = layer(hin)
        return hout

    # 全ての層を一括で初期化
    def reset_state(self):
        for layer in self:
            layer.reset_state()

    # 主に encoder と decoder 間の情報の受渡しや，beam searchの際に
    # 連続でLSTMを回せない時に一旦情報を保持するための関数
    def getAllLSTMStates(self):
        lstm_state_list_out = [0] * len(self) * 2
        for z in six.moves.range(len(self)):
            lstm_state_list_out[2 * z] = self[z].c
            lstm_state_list_out[2 * z + 1] = self[z].h
        # 扱いやすくするために，stackを使って一つの Chainer Variableにして返す
        return chaFunc.stack(lstm_state_list_out)

    # 用途としては，上のgetAllLSTMStatesで保存したものをセットし直すための関数
    def setAllLSTMStates(self, lstm_state_list_in):
        for z in six.moves.range(len(self)):
            self[z].c = lstm_state_list_in[2 * z]
            self[z].h = lstm_state_list_in[2 * z + 1]


# 組み込みのNStepLSTMを必要な形に修正したもの （cuDNNを使って高速化するため）
class NStepLSTMpp(chainer.ChainList):
    def __init__(self, n_layers,  # 層数
                 in_size,  # 一層目の入力の次元
                 out_size,  # 出力の次元(二層目以降の入力次元も同じ)
                 dropout_rate,
                 name="",
                 use_cudnn=True):
        weights = []
        direction = 1  # ここでは，からなず一方向ずつ構築するので1にする
        t_name = name
        if name != "":
            t_name = '%s_' % (name)

        for i in six.moves.range(n_layers):
            for di in six.moves.range(direction):
                weight = chainer.Link()
                for j in six.moves.range(8):
                    if i == 0 and j < 4:
                        w_in = in_size
                    elif i > 0 and j < 4:
                        w_in = out_size * direction
                    else:
                        w_in = out_size
                    weight.add_param('%sw%d' % (t_name, j), (out_size, w_in))
                    weight.add_param('%sb%d' % (t_name, j), (out_size,))
                    getattr(weight, '%sw%d' %
                            (t_name, j)).data[...] = np.random.normal(
                                0, np.sqrt(1. / w_in), (out_size, w_in))
                    getattr(weight, '%sb%d' % (t_name, j)).data[...] = 0
                weights.append(weight)

        super(NStepLSTMpp, self).__init__(*weights)

        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.use_cudnn = use_cudnn
        self.out_size = out_size
        self.direction = direction
        self.ws = [[getattr(w, '%sw0' % (t_name)),
                    getattr(w, '%sw1' % (t_name)),
                    getattr(w, '%sw2' % (t_name)),
                    getattr(w, '%sw3' % (t_name)),
                    getattr(w, '%sw4' % (t_name)),
                    getattr(w, '%sw5' % (t_name)),
                    getattr(w, '%sw6' % (t_name)),
                    getattr(w, '%sw7' % (t_name))] for w in self]
        self.bs = [[getattr(w, '%sb0' % (t_name)),
                    getattr(w, '%sb1' % (t_name)),
                    getattr(w, '%sb2' % (t_name)),
                    getattr(w, '%sb3' % (t_name)),
                    getattr(w, '%sb4' % (t_name)),
                    getattr(w, '%sb5' % (t_name)),
                    getattr(w, '%sb6' % (t_name)),
                    getattr(w, '%sb7' % (t_name))] for w in self]

    def init_hx(self, xs):
        hx_shape = self.n_layers * self.direction
        with cuda.get_device_from_id(self._device_id):
            hx = chainer.Variable(
                self.xp.zeros((hx_shape, xs.data.shape[1], self.out_size),
                              dtype=xs.dtype))
        return hx

    def __call__(self, hx, cx, xs):
        if hx is None:
            hx = self.init_hx(xs)
        if cx is None:
            cx = self.init_hx(xs)

        # hx, cx は (layer数, minibatch数，出力次元数)のtensor
        # xsは (系列長, minibatch数，出力次元数)のtensor
        # Note: chaFunc.n_step_lstm() は最初の入力層にはdropoutしない仕様
        hy, cy, ys = chaFunc.n_step_lstm(
            self.n_layers, self.dropout_rate, hx, cx, self.ws, self.bs, xs)
        # hy, cy は (layer数, minibatch数，出力次元数) で出てくる
        # ysは最終隠れ層だけなので，系列長のタプルで
        # 各要素が (minibatch数，出力次元数)
        # 扱いやすくするためにstackを使ってタプルを一つのchainer.Variableに変換
        # (系列長, minibatch数，出力次元数)のtensor
        hlist = chaFunc.stack(ys)
        return hy, cy, hlist


# LSTMの層の数を変数で決定したいので，層数が可変なことをこのクラスで吸収する
class NLayerCuLSTM(chainer.ChainList):
    def __init__(self, n_layers, eDim, hDim, name=""):
        layers = [0] * n_layers
        for z in six.moves.range(n_layers):
            if name != "":  # 名前を付ける
                t_name = '%s_L%d' % (name, z + 1)
            # 毎回一層分のNStepLSTMを作成
            if z == 0:
                tDim = eDim
            else:
                tDim = hDim
            # 手動で外でdropoutするのでここではrateを0に固定する
            layers[z] = NStepLSTMpp(1, tDim, hDim, dropout_rate=0.0,
                                    name=t_name)

        super(NLayerCuLSTM, self).__init__(*layers)

    # layre_numで指定された層をinput_state_listの長さ分回す
    def __call__(self, layer_num, input_state_list, dropout_rate):
        # Note: chaFunc.n_step_lstm() は最初の入力にはdropoutしない仕様なので，
        # 一層毎に手動で作った場合は手動でdropoutが必要
        if layer_num > 0:
            hin = chaFunc.dropout(input_state_list, ratio=dropout_rate)
        else:
            hin = input_state_list
        # layer_num層目の処理を一括で行う
        hy, cy, hout = self[layer_num](None, None, hin)
        return hy, cy, hout

#########################################
# TODO: とりあえず今の所は，後続の単語n_size文を予測するモードになっていて，複数の出力を読めるようにはなっていないので注意が必要!!
# n_size=1なら普通のoutputとして機能する


class DecInOutShareL(chainer.ChainList):
    def __init__(self, eDim, VocabSize, hDim=0, param_share=True, name=""):

        self.eDim = eDim
        self.hDim = hDim
        self.VocabSize = VocabSize
        self.param_share = param_share
        self.total_params = 0

        if hDim > 0:
            layers = [0] * 3  # 変換行列分
        else:
            layers = [0] * 2
        layers[0] = chaLink.EmbedID(self.VocabSize, self.eDim,
                                    ignore_label=-1)
        layers[0].W.name = name + "_Vocab_W"
        self.total_params += (self.eDim * self.VocabSize)

        if param_share:
            layers[1] = chaLink.Bias(shape=(self.VocabSize,))
            layers[1].b.name = name + "_output_b"
            self.total_params += (self.VocabSize)
        else:
            layers[1] = chaLink.Linear(self.eDim, self.VocabSize)
            layers[1].W.name = name + "_output_W"
            layers[1].b.name = name + "_output_b"
            self.total_params += (self.eDim * self.VocabSize
                                  + self.VocabSize)
        if hDim > 0:
            layers[2] = chaLink.Linear(self.hDim, self.eDim)
            layers[2].W.name = name + "_tr_W"
            layers[2].b.name = name + "_tr_b"
            self.total_params += (self.hDim * eDim + eDim)

        sys.stderr.write(
            ('# DecInOutShareL dim:{} {} {} | {} | total={}\n'
             .format(
                 self.eDim, self.VocabSize, self.hDim,
                 self.param_share, self.total_params
             )))
        super(DecInOutShareL, self).__init__(*layers)

    def to_gpu(self, gpu_num):
        for layer in self:
            layer.to_gpu(gpu_num)

    def getEmbedID(self, index_list):
        return self[0](index_list)

    def calcOutput(self, hIn):
        if self.hDim > 0:
            hIn = self[2](hIn)  # 次元数を合わせるため，一回変換
        if self.param_share:
            return chaFunc.linear(hIn, self[0].W, self[1].b)
        else:
            return self[1](hIn)

    def getTotalParams(self):
        return self.total_params

#########################################
# TODO: とりあえず今の所は，後続の単語n_size文を予測するモードになっていて，複数の出力を読めるようにはなっていないので注意が必要!!
# n_size=1なら普通のoutputとして機能する


class MultiOutputL(chainer.ChainList):
    def __init__(self, eDim, hDim, decVocabSize, param_share=True, name=""):

        self.total_params = 0
        self.eDim = eDim
        self.hDim = hDim
        self.decVocabSize = decVocabSize  # list
        self.param_share = param_share

        self.n_size = len(decVocabSize)
        assert self.n_size > 0, "ERROR"

        layers = [0] * self.n_size  # 層分の領域を確保
        for z in six.moves.range(self.n_size):
            if z == 0 and not self.param_share:
                t_param_share = False  # param_shareしない場合　
            else:
                t_param_share = True
            if self.eDim == self.hDim:  # 次元数が同じ時は hDimを渡さないことでパラメタ節約
                t_hDim = 0
            else:
                t_hDim = self.hDim

            layers[z] = DecInOutShareL(self.eDim, self.decVocabSize[z],
                                       hDim=t_hDim, param_share=t_param_share,
                                       name=name + "_MultiOutputL%d" % (z))
            self.total_params += layers[z].getTotalParams()

        sys.stderr.write(
            ('# MultiOutputL n_size:{} | dim:{} {} {} | total={}\n'
             .format(
                 self.n_size,
                 self.eDim, self.hDim, self.decVocabSize,
                 self.total_params
             )))
        super(MultiOutputL, self).__init__(*layers)

    def to_gpu(self, gpu_num):
        for layer in self:
            layer.to_gpu(gpu_num)

    def getInputEmbeddings(self, input_idx_list_mb, posi, args):
        xp = cuda.get_array_module(self[0][0].W.data)
        emb = self[posi].getEmbedID(
            chainer.Variable(xp.array(input_idx_list_mb)))
        return emb

    def calcLoss(self, train_mode, index, input_list,
                 cMBSize, decSent, dropout_rate, args):
        closs = 0
        hlast_dr = chaFunc.dropout(input_list[index], ratio=dropout_rate)
        correct = 0
        incorrect = 0
        proc = 0
        for z in six.moves.range(self.n_size):
            cLabel = decSent[z][index + 1]  # 正解は次の時刻なので，index+1を使用
            proc += (xp.count_nonzero(cLabel + 1))
            oVector_t = self[z].calcOutput(hlast_dr)
            closs += chaFunc.softmax_cross_entropy(
                oVector_t, cLabel, normalize=False)
            # TODO 全ての正解率の平均にしてもよい
            if (train_mode == 0 or args.doEvalAcc > 0):
                # 予測した単語のID配列 CuPy
                t_pred_arr = oVector_t.data.argmax(axis=1)
                t_correct = (
                    cLabel.size -
                    xp.count_nonzero(
                        cLabel -
                        t_pred_arr))
                # 予測不要の数から正解した数を引く # +1はbroadcast
                t_incorrect = xp.count_nonzero(cLabel + 1) - t_correct
                correct += t_correct
                incorrect += t_incorrect
        ###########
        closs /= (1.0 * self.n_size)  # スケールを合わせる為に，処理した数の平均にする
        ###########
        return closs, correct, incorrect, proc

    def getProbSingle(self, index, input_list, cMBSize, args):
        hlast_dr = chaFunc.dropout(input_list[index], ratio=0.0)
        oVector_t = self[0].calcOutput(hlast_dr)
        nextWordProb_a = -chaFunc.log_softmax(oVector_t.data).data
        return nextWordProb_a

    def getProbSingle2(self, index, input_list, cMBSize, args):
        hlast_dr = chaFunc.dropout(input_list[index], ratio=0.0)
        oVector_t = self[0].calcOutput(hlast_dr)
        nextWordProb_a = chaFunc.softmax(oVector_t.data).data
        return nextWordProb_a

# EncDecの本体


class EncoderDecoderAttention:
    def __init__(self, encoderVocab, decoderVocab, setting):
        self.encoderVocab = encoderVocab  # encoderの語彙
        self.decoderVocab = decoderVocab  # decoderの語彙
        # 語彙からIDを取得するための辞書
        self.index2encoderWord = {
            v: k for k, v in six.iteritems(
                self.encoderVocab)}  # 実際はなくてもいい
        self.index2decoderWord = [0] * len(self.decoderVocab)
        for i, f in enumerate(self.decoderVocab):
            self.index2decoderWord[i] = {
                v: k for k, v in six.iteritems(f)}  # decoderで利用
        ##########
        self.eDim = setting.eDim
        self.hDim = setting.hDim
        self.flag_dec_ifeed = setting.flag_dec_ifeed
        self.flag_enc_boseos = setting.flag_enc_boseos
        self.attn_mode = setting.attn_mode
        self.flag_merge_encfwbw = setting.flag_merge_encfwbw

        self.encVocabSize = len(encoderVocab)
        # self.decVocabSize = len(decoderVocab)
        self.decVocabSize = [0] * len(self.decoderVocab)
        for i, f in enumerate(self.decoderVocab):
            self.decVocabSize[i] = len(f)
        ###############
        self.n_layers = setting.n_layers

        # self.window_size = 1 #setting.window_size
        self.output_layer_type = setting.output_layer_type
        self.decEmbTying = setting.decEmbTying

    # encoder-docoderのネットワーク
    def initModel(self):
        sys.stderr.write(
            ('Vocab: enc={} dec={} embedDim: {}, hiddenDim: {}, '
             'n_layers: {} # [Params] dec inputfeed [{}] '
             '| use Enc BOS/EOS [{}] | attn mode [{}] '
             '| merge Enc FWBW [{}] '
             '| dec Emb Tying [{}]\n' .format(
                 self.encVocabSize,
                 self.decVocabSize,
                 self.eDim,
                 self.hDim,
                 self.n_layers,
                 self.flag_dec_ifeed,
                 self.flag_enc_boseos,
                 self.attn_mode,
                 self.flag_merge_encfwbw,
                 self.decEmbTying,
             )))
        self.model = chainer.Chain(
            # encoder embedding層
            encoderEmbed=chaLink.EmbedID(self.encVocabSize, self.eDim,
                                         ignore_label=-1),
        )
        # logに出力する際にわかりやすくするための名前付け なくてもよい
        self.model.encoderEmbed.W.name = "encoderEmbed_W"

        if self.output_layer_type == 0:
            # TODO: 今は評価時のbeam searchの処理を考えて0番目のdecVocabしか入力としては扱わない
            # TODO: 出力側の2番目以降の情報は，現在は学習時の正解としてのみ利用可 embeddingとしては使えない
            self.model.add_link(
                "decOutputL",
                MultiOutputL(self.eDim, self.hDim, self.decVocabSize,
                             param_share=self.decEmbTying,
                             name="decoderOutput"
                             ))
        else:
            assert 0, "ERROR"

        if self.flag_merge_encfwbw == 0:  # default
            self.model.add_link(
                "encLSTM_f",
                NStepLSTMpp(self.n_layers, self.eDim, self.hDim,
                            args.dropout_rate, name="encBiLSTMpp_fw"))
            self.model.add_link(
                "encLSTM_b",
                NStepLSTMpp(self.n_layers, self.eDim, self.hDim,
                            args.dropout_rate, name="encBiLSTMpp_bk"))
        elif self.flag_merge_encfwbw == 1:
            self.model.add_link(
                "encLSTM_f",
                NLayerCuLSTM(self.n_layers, self.eDim, self.hDim,
                             "encBiLSTM_fw"))
            self.model.add_link(
                "encLSTM_b",
                NLayerCuLSTM(self.n_layers, self.eDim, self.hDim,
                             "encBiLSTM_bk"))
        else:
            assert 0, "ERROR"

        # input feedの種類によって次元数が変わることに対応
        if self.flag_dec_ifeed == 0:  # inputfeedを使わない
            decLSTM_indim = self.eDim
        elif self.flag_dec_ifeed == 1:  # inputfeedを使う default
            decLSTM_indim = self.eDim + self.hDim
        # if   self.flag_dec_ifeed == 2: # inputEmbを使わない (debug用)
        #    decLSTM_indim = self.hDim
        else:
            assert 0, "ERROR"

        self.model.add_link(
            "decLSTM",
            NLayerLSTM(self.n_layers, decLSTM_indim, self.hDim, "decLSTM_fw"))

        # attentionの種類によってモデル構成が違うことに対応
        if self.attn_mode > 0:  # attn_mode == 1 or 2
            self.model.add_link(
                "attnIn_L1",
                chaLink.Linear(self.hDim, self.hDim, nobias=True))
            self.model.add_link(
                "attnOut_L2",
                chaLink.Linear(self.hDim + self.hDim, self.hDim, nobias=True))
            self.model.attnIn_L1.W.name = "attnIn_W"
            self.model.attnOut_L2.W.name = "attnOut_W"
        #
        if self.attn_mode == 2:  # attention == MLP
            self.model.add_link(
                "attnM",
                chaLink.Linear(self.hDim, self.hDim, nobias=True))
            self.model.add_link(
                "attnSum",
                chaLink.Linear(self.hDim, 1, nobias=True))
            self.model.attnM.W.name = "attnM_W"
            self.model.attnSum.W.name = "attnSum_W"

    #######################################
    # ネットワークの各構成要素をGPUのメモリに配置
    def setToGPUs(self, args):
        if args.gpu_enc >= 0 and args.gpu_dec >= 0:
            sys.stderr.write(
                '# Working on GPUs [gpu_enc=%d][gpu_dec=%d]\n' %
                (args.gpu_enc, args.gpu_dec))
            if not args.flag_emb_cpu:  # 指定があればCPU側のメモリ上に置く
                self.model.encoderEmbed.to_gpu(args.gpu_enc)
            self.model.encLSTM_f.to_gpu(args.gpu_enc)
            self.model.encLSTM_b.to_gpu(args.gpu_enc)

            # if not args.flag_emb_cpu:  # 指定があればCPU側のメモリ上に置く
            #    self.model.decoderEmbed.to_gpu(args.gpu_dec)
            self.model.decLSTM.to_gpu(args.gpu_dec)
            self.model.decOutputL.to_gpu(args.gpu_dec)

            if self.attn_mode > 0:
                self.model.attnIn_L1.to_gpu(args.gpu_dec)
                self.model.attnOut_L2.to_gpu(args.gpu_dec)
            if self.attn_mode == 2:
                self.model.attnSum.to_gpu(args.gpu_dec)
                self.model.attnM.to_gpu(args.gpu_dec)
        else:
            sys.stderr.write(
                '# NO GPUs [gpu_enc=%d][gpu_dec=%d]\n' %
                (args.gpu_enc, args.gpu_dec))

    #######################################
    def setInitAllParameters(self, optimizer, init_type="default",
                             init_scale=0.1):
        sys.stdout.write("############ Current Parameters BEGIN\n")
        self.printAllParameters(optimizer)
        sys.stdout.write("############ Current Parameters END\n")

        if init_type == "uniform":
            sys.stdout.write(
                "# initializer is [uniform] [%f]\n" %
                (init_scale))
            t_initializer = chainer.initializers.Uniform(init_scale)
            named_params = sorted(
                optimizer.target.namedparams(),
                key=lambda x: x[0])
            for n, p in named_params:
                with cuda.get_device(p.data):
                    p.copydata(chainer.Parameter(
                        t_initializer, p.data.shape))
        elif init_type == "normal":
            sys.stdout.write("# initializer is [normal] [%f]\n" % (init_scale))
            t_initializer = chainer.initializers.Normal(init_scale)
            named_params = sorted(
                optimizer.target.namedparams(),
                key=lambda x: x[0])
            for n, p in named_params:
                with cuda.get_device(p.data):
                    p.copydata(chainer.Parameter(
                        t_initializer, p.data.shape))
        else:  # "default"
            sys.stdout.write(
                "# initializer is [defalit] [%f]\n" %
                (init_scale))
            named_params = sorted(
                optimizer.target.namedparams(),
                key=lambda x: x[0])
            for n, p in named_params:
                with cuda.get_device(p.data):
                    p.data *= args.init_scale
        self.printAllParameters(optimizer, init_type, init_scale)
        return 0

    def printAllParameters(self, optimizer, init_type="***", init_scale=1.0):
        total_norm = 0
        total_param = 0
        named_params = sorted(
            optimizer.target.namedparams(),
            key=lambda x: x[0])
        for n, p in named_params:
            t_norm = _sum_sqnorm(p.data)
            sys.stdout.write(
                '### {} {} {} {} {}\n'.format(
                    p.name, p.data.ndim, p.data.shape, p.data.size, t_norm))
            total_norm += t_norm
            total_param += p.data.size
        with cuda.get_device(total_norm):
            sys.stdout.write(
                '# param size= [{}] norm = [{}] scale=[{}, {}]\n'.format(
                    total_param, self.model.xp.sqrt(total_norm),
                    init_type, init_scale))

    ###############################################
    # 学習済みのw2vデータを読み込む
    ###############################################
    def readWord2vecResult(self, init_emb_by_w2v_file, init_emb_by_w2v_mode):
        srcVCount = 0
        trgVCount = [0] * len(self.decVocabSize)
        i = 0

        for t_file in init_emb_by_w2v_file.split(':'):
            sys.stdout.write('# W2V file: [%s]\n' % (t_file))
            for i, line in enumerate(codecs.open(
                    t_file, "r", "utf-8", "ignore")):
                if i == 0:   # 一行目はステータス行なので，とばす
                    status = line.strip().split(' ')
                    vec = np.array(status, dtype=np.int32)
                    sys.stdout.write(
                        '# W2V file: [{}] [{}][{}][{}] | mode:[{}]\n'.format(
                            t_file, len(vec), vec[0], vec[1],
                            init_emb_by_w2v_mode))
                    continue
                else:
                    lineList = line.strip().split(' ')
                    word = lineList[0]
                    vec = np.array(lineList[1:], dtype=np.float32)
                    if i < 10:
                        sys.stdout.write(
                            '# W2V file: [%d][%d]\n' % (i, len(lineList)))

                    if (init_emb_by_w2v_mode == 0 or
                            init_emb_by_w2v_mode == 2) and \
                            word in self.encoderVocab:  # encoderの語彙
                        self.model.encoderEmbed.W.data[
                            self.encoderVocab[word]] = xp.array(vec)
                        srcVCount += 1
                    if (init_emb_by_w2v_mode == 1 or
                            init_emb_by_w2v_mode == 2):
                        for z in six.moves.range(len(self.decoderVocab)):
                            if word in self.decoderVocab[z]:  # decoderの語彙
                                self.model.decOutputL[z][0].W.data[
                                    self.decoderVocab[z][word]] = xp.array(vec)
                                trgVCount[z] += 1

            if 1:
                sys.stdout.write(
                    '# W2V file: [%s]  DONE %d lines '
                    '| encVCount %d %d %.2f\n' % (
                        t_file, i,
                        srcVCount, len(self.encoderVocab),
                        float(100.0 * srcVCount / len(self.encoderVocab)),
                    ))
            for z in six.moves.range(len(self.decVocabSize)):
                sys.stdout.write(
                    '# W2V file: [%s]  DONE %d lines '
                    '| decVCount [%d] %d %d %.2f\n' %
                    (t_file, i, z, trgVCount[z], len(
                        self.decoderVocab[z]), float(
                        100.0 * trgVCount[z] / len(
                            self.decoderVocab[z])), ))
        return i

    ###############################################
    # 情報を保持するためだけのクラス 主に 細切れにbackwardするための用途
    class encInfoObject:
        def __init__(self, finalHiddenVars, finalLSTMVars,
                     enc4BKWOrig, enc4BKWCopy, encLen, cMBSize):
            self.attnList = finalHiddenVars
            self.lstmVars = finalLSTMVars
            self.enc4BKWOrig = enc4BKWOrig  # 計算グラフを分断した際の元のVariable
            self.enc4BKWCopy = enc4BKWCopy  # 計算グラフを分断した際の新しく作成したVariable
            self.encLen = encLen
            self.cMBSize = cMBSize
    ###############################################

    def getEncoderInputEmbeddings(self, input_idx_list_mb, args):
        # 一文一括でembeddingを取得  この方が効率が良い？
        if args.flag_emb_cpu and args.gpu_enc >= 0:
            encEmbList = chaFunc.copy(
                self.model.encoderEmbed(chainer.Variable(
                    np.array(input_idx_list_mb, dtype=np.int32).T
                )))
        else:
            xp = cuda.get_array_module(self.model.encoderEmbed.W.data)
            # 3
            # cMBSize = len(input_idx_list_mb)
            # encLen = len(input_idx_list_mb[0])
            # fsize = len(input_idx_list_mb[0][0])
            # 3
            encEmbList = self.model.encoderEmbed(
                chainer.Variable(
                    xp.transpose(xp.array(input_idx_list_mb), axes=(1, 0, 2)))
            )
            # sys.stdout.write('#2 {} | {} {}\n'.format(encEmbList.shape,
            # cMBSize, encLen))
            encEmbList = chaFunc.sum(encEmbList, axis=2)
            # sys.stdout.write('#3 {} | {} {}\n'.format(encEmbList.shape,
            # cMBSize, encLen))
        return encEmbList

    # encoder側の入力を処理する関数
    def encodeSentenceFWD(self, train_mode, sentence, args, dropout_rate):
        if args.gpu_enc != args.gpu_dec:  # encとdecが別GPUの場合
            chainer.cuda.get_device(args.gpu_enc).use()
        # encLen = len(sentence)  # 文長
        # cMBSize = len(sentence[0])  # minibatch size
        encLen = len(sentence[0])  # 文長
        cMBSize = len(sentence)  # minibatch size

        # 一文一括でembeddingを取得  この方が効率が良い？
        encEmbList = self.getEncoderInputEmbeddings(sentence, args)

        # flag_train = (train_mode > 0)
        lstmVars = [0] * self.n_layers * 2
        if self.flag_merge_encfwbw == 0:  # fwとbwは途中で混ぜない最後で混ぜる
            hyf, cyf, fwHout = self.model.encLSTM_f(
                None, None, encEmbList)  # 前向き
            hyb, cyb, bkHout = self.model.encLSTM_b(
                None, None, encEmbList[::-1])  # 後向き
            for z in six.moves.range(self.n_layers):
                lstmVars[2 * z] = cyf[z] + cyb[z]
                lstmVars[2 * z + 1] = hyf[z] + hyb[z]
        elif self.flag_merge_encfwbw == 1:  # fwとbwを一層毎に混ぜる
            sp = (cMBSize, self.hDim)
            for z in six.moves.range(self.n_layers):
                if z == 0:  # 一層目 embeddingを使う
                    biH = encEmbList
                else:  # 二層目以降 前層の出力を使う
                    # 加算をするためにbkHoutの逆順をもとの順序に戻す
                    biH = fwHout + bkHout[::-1]
                # z層目前向き
                hyf, cyf, fwHout = self.model.encLSTM_f(
                    z, biH, dropout_rate)
                # z層目後ろ向き
                hyb, cyb, bkHout = self.model.encLSTM_b(
                    z, biH[::-1], dropout_rate)
                # それぞれの階層の隠れ状態およびメモリセルをデコーダに
                # 渡すために保持
                lstmVars[2 * z] = chaFunc.reshape(cyf[0] + cyb[0], sp)
                lstmVars[2 * z + 1] = chaFunc.reshape(hyf[0] + hyb[0], sp)
        else:
            assert 0, "ERROR"

        # 最終隠れ層
        if self.flag_enc_boseos == 0:  # default
            # fwHoutを[:,]しないとエラーになる？
            biHiddenStack = fwHout[:, ] + bkHout[::-1]
        elif self.flag_enc_boseos == 1:
            bkHout2 = bkHout[::-1]  # 逆順を戻す
            biHiddenStack = fwHout[1:encLen - 1, ] + bkHout2[1:encLen - 1, ]
            # BOS, EOS分を短くする TODO おそらく長さ0のものが入るとエラー
            encLen -= 2
        else:
            assert 0, "ERROR"
        # (encの単語数, minibatchの数, 隠れ層の次元)
        #    => (minibatchの数, encの単語数, 隠れ層の次元)に変更
        biHiddenStackSW01 = chaFunc.swapaxes(biHiddenStack, 0, 1)
        # 各LSTMの最終状態を取得して，decoderのLSTMの初期状態を作成
        lstmVars = chaFunc.stack(lstmVars)

        if train_mode > 0:  # backwardを効率的にするためにやや面倒なことをする
            enc4BKWOrig = chaFunc.concat(
                (biHiddenStackSW01, chaFunc.swapaxes(
                    lstmVars, 0, 1)), axis=1)
            # decが別GPUの場合にデバイス間コピー
            if args.gpu_enc != args.gpu_dec:
                # backwardするので，to_gpu()ではなくcopy()
                enc4BKWOrig = chaFunc.copy(enc4BKWOrig, args.gpu_dec)
            # ここから先は dec側のGPUのメモリにはりつく
            # ここで計算グラフを分断
            enc4BKWCopy = chainer.Variable(enc4BKWOrig.data)
            # LSTMの隠れ状態の部分
            finalHiddenVars = enc4BKWCopy[:, 0:encLen, ]
            # 3行前でswapしたのを元に戻す
            finalLSTMVars = chaFunc.swapaxes(enc4BKWCopy[:, encLen:, ], 0, 1)
        else:  # テスト時は backwardいらないので，なにもせずに受渡し
            enc4BKWOrig = None  # backward用の構造はいらないので．．．
            enc4BKWCopy = None  # backward用の構造はいらないので．．．
            finalHiddenVars = biHiddenStackSW01
            finalLSTMVars = lstmVars
            # decが別GPUの場合にデバイス間コピー
            if args.gpu_enc != args.gpu_dec:
                # backwardしないので copy()は不要？ to_gpu()で十分
                finalHiddenVars.to_gpu(args.gpu_dec)
                finalLSTMVars.to_gpu(args.gpu_dec)

        # encoderの情報をencInfoObjectに集約して返す
        retO = self.encInfoObject(finalHiddenVars, finalLSTMVars,
                                  enc4BKWOrig, enc4BKWCopy, encLen, cMBSize)
        return retO

    def encodeSentenceBKWD(self, encInfo):
        encInfo.enc4BKWOrig.addgrad(encInfo.enc4BKWCopy)
        encInfo.enc4BKWOrig.backward()

    def prepareDecoder(self, encInfo):
        self.model.decLSTM.reset_state()
        if self.attn_mode == 0:
            aList = None
        elif self.attn_mode == 1:
            aList = encInfo.attnList
        elif self.attn_mode == 2:
            aList = self.model.attnM(
                chaFunc.reshape(encInfo.attnList,
                                (encInfo.cMBSize * encInfo.encLen, self.hDim)))
            # TODO: 効率が悪いのでencoder側に移動したい
        else:
            assert 0, "ERROR"
        xp = cuda.get_array_module(encInfo.lstmVars[0].data)
        finalHS = chainer.Variable(
            xp.zeros(
                encInfo.lstmVars[0].data.shape,
                dtype=xp.float32))  # 最初のinput_feedは0で初期化
        return aList, finalHS

    ############################
    def trainOneMiniBatch(self, train_mode, decSent, encInfo,
                          args, dropout_rate):
        if args.gpu_enc != args.gpu_dec:  # encとdecが別GPUの場合
            chainer.cuda.get_device(args.gpu_dec).use()
        cMBSize = encInfo.cMBSize
        aList, finalHS = self.prepareDecoder(encInfo)

        xp = cuda.get_array_module(encInfo.lstmVars[0].data)
        # total_loss = chainer.Variable(xp.zeros((), dtype=xp.float32))
        total_loss_val = 0  # float
        correct = 0
        incorrect = 0
        proc = 0
        decoder_proc = len(decSent[0]) - 1  # ここで処理するdecoder側の単語数

        #######################################################################
        # 1, decoder側の入力単語embeddingsをまとめて取得
        decEmbListOrig = self.model.decOutputL.getInputEmbeddings(
            decSent[0][:decoder_proc], 0, args)
        decEmbListCopy = chainer.Variable(decEmbListOrig.data)  # ここで切断
        decSent = xp.array(decSent)  # GPU上に移動
        #######################################################################
        # 2, decoder側のRNN部分を計算
        h4_list_orig = [0] * decoder_proc
        h4_list_copy = [0] * decoder_proc
        lstm_states_list_orig = [0] * decoder_proc
        lstm_states_list_copy = [0] * decoder_proc
        for index in six.moves.range(decoder_proc):  # decoder_len -1
            if index == 0:
                t_lstm_states = encInfo.lstmVars
                t_finalHS = finalHS
            else:
                t_lstm_states = lstm_states_list_copy[index - 1]
                t_finalHS = h4_list_copy[index - 1]
            # decoder LSTMを一回ぶん計算
            hOut, lstm_states = self.processDecLSTMOneStep(
                decEmbListCopy[index], t_lstm_states,
                t_finalHS, args, dropout_rate)
            # lstmの接続をここで分断
            lstm_states_list_orig[index] = lstm_states
            lstm_states_list_copy[index] = chainer.Variable(lstm_states.data)
            # Note: 計算グラフを分断する関係で，上で取得したhOutではなく
            # 全く同じ情報をlstm_statesの方から取得する
            hOut = lstm_states_list_copy[index][-1]

            # attentionありの場合 contextベクトルを計算
            finalHS = self.calcAttention(hOut, encInfo.attnList, aList,
                                         encInfo.encLen, cMBSize, args)
            # 最終隠れ層でも計算グラフを分離
            h4_list_orig[index] = finalHS
            h4_list_copy[index] = chainer.Variable(finalHS.data)
        #######################################################################
        # 3, output(softmax)層の計算
        for index in reversed(six.moves.range(decoder_proc)):
            ###################
            closs, t_correct, t_incorrect, t_proc = \
                self.model.decOutputL.calcLoss(
                    train_mode, index, h4_list_copy,
                    cMBSize, decSent, dropout_rate, args)
            proc += t_proc
            correct += t_correct
            incorrect += t_incorrect
            total_loss_val += closs.data * cMBSize * \
                len(decSent)  # clossはdecの出力種類数で割っているので，ここで辻褄を合わせる
            if train_mode > 0:  # 学習データのみ backward する
                closs.backward()  # 最終出力層から，最終隠れ層までbackward
                # 最終隠れ層まで戻ってきたgradを加算
                # ここでinputfeedのgradは一つ前のループで加算されている仮定
                h4_list_orig[index].addgrad(h4_list_copy[index])
                h4_list_orig[index].backward()  # backward計算
                # lstmの状態で分断したところのgradを戻す
                lstm_states_list_orig[index].addgrad(
                    lstm_states_list_copy[index])
                lstm_states_list_orig[index].backward()
            # 使い終わったデータの領域を解放
            # 実際にここまでやる必要があるかは不明
            del closs
            del lstm_states_list_orig[index]
            del lstm_states_list_copy[index]
            del h4_list_orig[index]
            del h4_list_copy[index]

        ####
        if train_mode > 0:  # 学習時のみ backward する
            decEmbListOrig.addgrad(decEmbListCopy)
            decEmbListOrig.backward()        # decoderのembeddingのbackward実行
            self.encodeSentenceBKWD(encInfo)  # encoderのbackward実行

        return total_loss_val, (correct, incorrect, decoder_proc, proc)

    # decoder LSTMの計算
    def processDecLSTMOneStep(self, decInputEmb, lstm_states_in,
                              finalHS, args, dropout_rate):
        # 1, RNN層を隠れ層の値をセット
        # （beam searchへの対応のため毎回必ずセットする）
        self.model.decLSTM.setAllLSTMStates(lstm_states_in)
        # 2, 単語埋め込みの取得とinput feedの処理
        if self.flag_dec_ifeed == 0:  # inputfeedを使わない
            wenbed = decInputEmb
        elif self.flag_dec_ifeed == 1:  # inputfeedを使う (default)
            wenbed = chaFunc.concat((finalHS, decInputEmb))
        # elif self.flag_dec_ifeed == 2: # decInputEmbを使わない (debug用)
        #    wenbed = finalHS
        else:
            assert 0, "ERROR"
        # 3， N層分のRNN層を一括で計算
        h1 = self.model.decLSTM.processOneStepForward(
            wenbed, args, dropout_rate)
        # 4, 次の時刻の計算のためにLSTMの隠れ層を取得
        lstm_states_out = self.model.decLSTM.getAllLSTMStates()
        return h1, lstm_states_out

    # attentionの計算
    def calcAttention(self, h1, hList, aList, encLen, cMBSize, args):
        # attention使わないなら入力された最終隠れ層h1を返す
        if self.attn_mode == 0:
            return h1
        # 1, attention計算のための準備
        target1 = self.model.attnIn_L1(h1)  # まず一回変換
        # (cMBSize, self.hDim) => (cMBSize, 1, self.hDim)
        target2 = chaFunc.expand_dims(target1, axis=1)
        # (cMBSize, 1, self.hDim) => (cMBSize, encLen, self.hDim)
        target3 = chaFunc.broadcast_to(target2, (cMBSize, encLen, self.hDim))
        # target3 = chaFunc.broadcast_to(chaFunc.reshape(
        #    target1, (cMBSize, 1, self.hDim)), (cMBSize, encLen, self.hDim))
        # 2, attentionの種類に従って計算
        if self.attn_mode == 1:  # bilinear
            # bilinear系のattentionの場合は，hList1 == hList2 である
            # shape: (cMBSize, encLen)
            aval = chaFunc.sum(target3 * aList, axis=2)
        elif self.attn_mode == 2:  # MLP
            # attnSum に通すために変形
            t1 = chaFunc.reshape(target3, (cMBSize * encLen, self.hDim))
            # (cMBSize*encLen, self.hDim) => (cMBSize*encLen, 1)
            t2 = self.model.attnSum(chaFunc.tanh(t1 + aList))
            # shape: (cMBSize, encLen)
            aval = chaFunc.reshape(t2, (cMBSize, encLen))
            # aval = chaFunc.reshape(self.model.attnSum(
            #    chaFunc.tanh(t1 + aList)), (cMBSize, encLen))
        else:
            assert 0, "ERROR"
        # 3, softmaxを求める
        cAttn1 = chaFunc.softmax(aval)   # (cMBSize, encLen)
        # 4, attentionの重みを使ってcontext vectorを作成するところ
        # (cMBSize, encLen) => (cMBSize, 1, encLen)
        cAttn2 = chaFunc.expand_dims(cAttn1, axis=1)
        # (1, encLen) x (encLen, hDim) の行列演算(matmul)をcMBSize回繰り返す
        #     => (cMBSize, 1, hDim)
        cAttn3 = chaFunc.batch_matmul(cAttn2, hList)
        # cAttn3 = chaFunc.batch_matmul(chaFunc.reshape(
        #    cAttn1, (cMBSize, 1, encLen)), hList)
        # axis=1の次元1になっているところを削除
        context = chaFunc.reshape(cAttn3, (cMBSize, self.hDim))
        # 4, attentionの重みを使ってcontext vectorを作成するところ
        # こっちのやり方でも可
        # (cMBSize, scrLen) => (cMBSize, scrLen, hDim)
        # cAttn2 = chaFunc.reshape(cAttn1, (cMBSize, encLen, 1))
        # (cMBSize, scrLen) => (cMBSize, scrLen, hDim)
        # cAttn3 = chaFunc.broadcast_to(cAttn2, (cMBSize, encLen, self.hDim))
        # 重み付き和を計算 (cMBSize, encLen, hDim)
        #     => (cMBSize, hDim)  # axis=1 がなくなる
        # context = chaFunc.sum(aList * cAttn3, axis=1)
        # 6, attention時の最終隠れ層の計算
        c1 = chaFunc.concat((h1, context))
        c2 = self.model.attnOut_L2(c1)
        finalH = chaFunc.tanh(c2)
        # finalH = chaFunc.tanh(self.model.attnOut_L2(
        #    chaFunc.concat((h1, context))))
        return finalH  # context

    # 出力層の計算
    # def generateWord(self, h4, encLen, cMBSize, args, dropout_rate):
    #     oVector = self.model.decOutputL(
    #         chaFunc.dropout(h4, ratio=dropout_rate))
    #     return oVector


########################################################
# データを読み込んだりするための関数をまとめたもの
class PrepareData:
    def __init__(self, setting):
        self.flag_enc_boseos = setting.flag_enc_boseos

    ################################################
    def readVocab(self, vocabFile):  # 学習時にのみ呼ばれる予定
        d = {}
        d.setdefault('<unk>', len(d))  # 0番目 固定
        sys.stdout.write('# Vocab: add <unk> | id={}\n'.format(d['<unk>']))
        d.setdefault('<s>', len(d))   # 1番目 固定
        sys.stdout.write('# Vocab: add <s>   | id={}\n'.format(d['<s>']))
        d.setdefault('</s>', len(d))  # 2番目 固定
        sys.stdout.write('# Vocab: add </s>  | id={}\n'.format(d['</s>']))

        # TODO: codecsでないとエラーが出る環境がある？ 要調査 不要ならioにしたい
        with io.open(vocabFile, encoding='utf-8') as f:
            # with codecs.open(vocabFile, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                word, freq = line.split('\t')  # 基本的にtab区切りを想定
                # word, freq = line.split(' ')  # スペース区切りはこちらに変更
                if word == "<unk>":
                    continue
                elif word == "<s>":
                    continue
                elif word == "</s>":
                    continue
                d.setdefault(word, len(d))
        return d

    def sentence2index(self, sentence, word2indexDict, input_side=False):
        indexList = [word2indexDict[word] if word in word2indexDict
                     else word2indexDict['<unk>']
                     for word in sentence.split(' ')]
        # encoder側でかつ，<s>と</s>を使わない設定の場合
        if input_side and self.flag_enc_boseos == 0:
            return indexList
        else:  # 通常はこちら
            return ([word2indexDict['<s>']] +
                    indexList + [word2indexDict['</s>']])

    def inputsentence2index(self, sentence, word2indexDict, input_side=False):
        token_list = sentence.split(' ')  # 単語単位に分割
        indexList = [0] * len(token_list)
        for i, token_idxs in enumerate(token_list):
            widxlst = [word2indexDict[word] if word in
                       word2indexDict else word2indexDict['<unk>']
                       for word in token_idxs.split('|||')]
            indexList[i] = widxlst
        maxfsize = max([len(widxs) for widxs in indexList])
        indexList = [widxs + [-1] * (maxfsize - len(widxs))
                     for widxs in indexList]
        # encoder側でかつ，<s>と</s>を使わない設定の場合
        if input_side and self.flag_enc_boseos == 0:
            return indexList
        else:  # 通常はこちら
            return ([[word2indexDict['<s>']] + [-1] * (maxfsize - 1)] +
                    indexList + [[word2indexDict['</s>']] + [-1] * (maxfsize
                                                                    - 1)])

    def makeSentenceLenDict(self, fileName, word2indexDict, input_side=False):
        if input_side:
            d = collections.defaultdict(list)
        else:
            d = {}
        sentenceNum = 0
        sampleNum = 0
        maxLen = 0
        unk_count = 0
        bos_count = 0
        eos_count = 0

        idx_bos = word2indexDict['<s>']
        idx_eos = word2indexDict['</s>']
        idx_unk = word2indexDict['<unk>']

        # ここで全てのデータを読み込む
        # TODO: codecsでないとエラーが出る環境がある？ 要調査 不要ならioにしたい
        with io.open(fileName, encoding='utf-8') as f:
            # with codecs.open(fileName, encoding='utf-8') as f:
            for sntNum, snt in enumerate(f):  # ここで全てのデータを読み込む
                snt = snt.strip()
                if input_side:
                    indexList = self.inputsentence2index(
                        snt, word2indexDict, input_side=input_side)
                    # input側 ここで長さ毎でまとめたリストを作成する
                    # 値は文番号と文そのもののペア
                    d[len(indexList)].append((sntNum, indexList))
                    # sys.stdout.write('# {} {}\n'.format(
                    #    sntNum, indexList))
                else:
                    indexList = self.sentence2index(
                        snt, word2indexDict, input_side=input_side)
                    d[sntNum] = indexList  # decoder側 文の番号をキーとしたハッシュ
                sampleNum += len(indexList)
                unk_count += indexList.count(idx_unk)
                bos_count += indexList.count(idx_bos)
                eos_count += indexList.count(idx_eos)
                sentenceNum += 1
                maxLen = max(maxLen, len(indexList))

        sys.stdout.write(
            '# data sent: %10d  sample: %10d maxlen: %10d '
            '| unk_rate %.2f %10d/%10d '
            '| #bos=%10d #eos=%10d\n' %
            (sentenceNum,
             sampleNum,
             maxLen,
             100.0 *
             unk_count /
             sampleNum,
             unk_count,
             sampleNum,
             bos_count,
             eos_count))
        return d

    def makeBatch4Train(self, encSentLenDict, decSentLenDict,
                        batch_size=1, shuffle_flag=True):
        encSentDividedBatch = []
        for length, encSentList in six.iteritems(encSentLenDict):
            random.shuffle(encSentList)  # ここで同じencLenのデータをshuffle
            iter2 = six.moves.range(0, len(encSentList), batch_size)
            encSentDividedBatch.extend(
                [encSentList[_:_ + batch_size] for _ in iter2])
        if shuffle_flag is True:
            # encLenの長さでまとめたものをシャッフルする
            random.shuffle(encSentDividedBatch)
        else:
            sys.stderr.write(
                ('# NO shuffle: descending order based on '
                 'encoder sentence length\n'))

        encSentBatch = []
        decSentBatch = []
        # shuffleなしの場合にencoderの長い方から順番に生成
        for batch in encSentDividedBatch[::-1]:
            # encSentBatch.append(
            #    np.array([encSent for sntNum, encSent in batch],
            #             dtype=np.int32).T)
            maxEncFsize = max([len(encSent[0])  # 最初の単語のサイズがわかればよい
                               for sntNum, encSent in batch])
            t_encSent = [0] * len(batch)
            i = 0
            for sntNum, encSent in batch:
                t_encSent[i] = [widxs + [-1] *
                                (maxEncFsize - len(widxs))
                                for widxs in encSent]
                i = i + 1
            encSentBatch.append(t_encSent)
            # 全てのデータに対して要素数を比較してもっとも大きい値を取得
            decSentTypeSize = len(decSentLenDict)
            maxDecoderLength = max([len(decSentLenDict[z][sntNum])
                                    for sntNum, encSent in batch
                                    for z in six.moves.range(decSentTypeSize)])

            t_decSentBatch = [0] * decSentTypeSize
            for z in six.moves.range(decSentTypeSize):
                t_decSentBatch[z] = np.array(
                    [decSentLenDict[z][sntNum] + [-1] *
                     (maxDecoderLength - len(decSentLenDict[z][sntNum]))
                     for sntNum, encSent in batch], dtype=np.int32).T
            # decSentBatchの構成は，axis=0が書くデータに相当するので，必ず第一軸でスライスして使う必要がある
            decSentBatch.append(t_decSentBatch)

            # TODO
        ######
        return list(six.moves.zip(encSentBatch, decSentBatch))


# 主に学習時の状況を表示するための情報を保持するクラス
class TrainProcInfo:
    def __init__(self):
        self.lossVal = 0
        self.instanceNum = 0
        self.corTot = 0
        self.incorTot = 0
        self.batchCount = 0
        self.trainsizeTot = 0
        self.procTot = 0

        self.gnorm = 0
        self.gnormLimit = 0
        self.pnorm = 0

        self.encMaxLen = 0
        self.decMaxLen = 0

    # 途中経過を標示するための情報取得するルーチン
    def print_strings(self, train_mode, epoch, cMBSize, encLen, decLen,
                      start_time, args):
        with cuda.get_device(self.lossVal):
            msg0 = 'Epoch: %3d | LL: %9.6f PPL: %10.4f' % (
                epoch, float(self.lossVal / max(1, self.procTot)),
                math.exp(min(10, float(self.lossVal / max(1, self.procTot)))))
            msg1 = '| gN: %8.4f %8.4f %8.4f' % (
                self.gnorm, self.gnormLimit, self.pnorm)
            dt = self.corTot + self.incorTot
            msg2 = '| acc: %6.2f %8d %8d ' % (
                float(100.0 * self.corTot / max(1, dt)),
                self.corTot, self.incorTot)
            msg3 = '| tot: %8d proc: %8d | num: %8d %6d %6d ' % (
                self.trainsizeTot, self.procTot, self.instanceNum,
                self.encMaxLen, self.decMaxLen)
            msg4 = '| MB: %4d %6d %4d %4d | Time: %10.4f' % (
                cMBSize, self.batchCount,
                encLen, decLen, time.time() - start_time)
            # dev.dataのときは必ず評価，学習データのときはオプションに従う
            if train_mode == 0:
                msgA = '%s %s %s %s' % (msg0, msg2, msg3, msg4)
            elif args.doEvalAcc > 0:
                msgA = '%s %s %s %s %s' % (msg0, msg1, msg2, msg3, msg4)
            else:
                msgA = '%s %s %s %s' % (msg0, msg1, msg3, msg4)
            return msgA


# 学習用のサブルーチン
def train_model_sub(train_mode, epoch, tData, EncDecAtt, optimizer,
                    clip_obj, start_time, args):
    if 1:  # 並列処理のコードとインデントを揃えるため．．．
        #####################
        tInfo = TrainProcInfo()
        prnCnt = 0
        #####################
        if train_mode > 0:  # train
            dropout_rate = args.dropout_rate
        else:              # dev
            dropout_rate = 0
        #####################
        if train_mode > 0:  # train
            sys.stderr.write(
                ('# TRAIN epoch {} drop rate={} | CHAINER CONFIG  [{}] \n'
                 .format(epoch, dropout_rate,
                         chainer.global_config.__dict__)))
        else:              # dev
            sys.stderr.write(
                ('# DEV.  epoch {} drop rate={} | CHAINER CONFIG  [{}] \n'
                 .format(epoch, dropout_rate,
                         chainer.global_config.__dict__)))
        #####################
        # メインループ
        for encSent, decSent in tData:
            # if 1:
            try:
                ###########################
                if train_mode > 0:  # train
                    EncDecAtt.model.cleargrads()  # パラメタ更新のためにgrad初期化
                ###########################
                encInfo = EncDecAtt.encodeSentenceFWD(
                    train_mode, encSent, args, dropout_rate)
                loss_stat, acc_stat = EncDecAtt.trainOneMiniBatch(
                    train_mode, decSent, encInfo, args, dropout_rate)
                ###########################
                # mini batch のiサイズは毎回違うので取得
                cMBSize = encInfo.cMBSize
                encLen = len(encSent)
                decLen = len(decSent[0])  # 複数あるので
                tInfo.instanceNum += cMBSize  # 文数を数える
                tInfo.batchCount += 1  # minibatchで何回処理したか
                tInfo.corTot += acc_stat[0]
                tInfo.incorTot += acc_stat[1]
                tInfo.trainsizeTot += acc_stat[2]
                tInfo.procTot += acc_stat[3]
                # 強制的にGPUからCPUに値を移すため floatを利用
                tInfo.lossVal += float(loss_stat)
                ###########################
                if train_mode > 0:
                    optimizer.update()  # ここでパラメタ更新
                    ###########################
                    tInfo.gnorm = clip_obj.norm_orig
                    tInfo.gnormLimit = clip_obj.threshold
                    if prnCnt == 100:
                        # TODO 処理が重いので実行回数を減らす ロが不要ならいらない
                        xp = cuda.get_array_module(encInfo.lstmVars[0].data)
                        tInfo.pnorm = float(
                            xp.sqrt(_sum_sqnorm(
                                [p.data for p in optimizer.target.params()])))
                ####################
                del encInfo
                ###################
                tInfo.encMaxLen = max(encLen * cMBSize, tInfo.encMaxLen)
                tInfo.decMaxLen = max(decLen * cMBSize, tInfo.decMaxLen)
                ###################
                if args.verbose == 0:
                    pass  # 途中結果は表示しない
                else:
                    msgA = tInfo.print_strings(
                        train_mode, epoch, cMBSize, encLen, decLen,
                        start_time, args)
                    if train_mode > 0 and prnCnt >= 100:
                        if args.verbose > 1:
                            sys.stdout.write('\r')
                        sys.stdout.write('%s\n' % (msgA))
                        prnCnt = 0
                    elif args.verbose > 2:
                        sys.stderr.write('\n%s' % (msgA))
                    elif args.verbose > 1:
                        sys.stderr.write('\r%s' % (msgA))
                ###################
                prnCnt += 1
            except Exception as e:
                # メモリエラーなどが発生しても処理を終了せずに
                # そのサンプルをスキップして次に進める
                flag = 0
                if args.gpu_enc >= 0 and args.gpu_dec >= 0:
                    import cupy
                    if isinstance(e, cupy.cuda.runtime.CUDARuntimeError):
                        cMBSize = len(encSent[0])
                        encLen = len(encSent)
                        decLen = len(decSent[0])
                        sys.stdout.write(
                            ('\r# GPU Memory Error? Skip! {} | enc={} dec={} '
                             'mbs={} total={} | {}\n'.format(
                                 tInfo.batchCount, encLen, decLen, cMBSize,
                                 (encLen + decLen) * cMBSize, type(e))))
                        sys.stdout.flush()
                        flag = 1
                if flag == 0:
                    sys.stdout.write(
                        ('\r# Fatal Error? {} | {} | {}\n'.format(
                            tInfo.batchCount, type(e), e.args)))
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    sys.exit(255)
        ###########################
        return tInfo
#########################################


#########################################
# optimizerの準備
def setOptimizer(args, EncDecAtt):
    # optimizerを構築
    if args.optimizer == 'SGD':
        optimizer = chaOpt.SGD(lr=args.lrate)
        sys.stdout.write(
            '# SET Learning %s: initial learning rate: %e\n' %
            (args.optimizer, optimizer.lr))
    elif args.optimizer == 'Adam':
        # assert 0, "Currently Adam is not supported for asynchronous update"
        optimizer = chaOpt.Adam(alpha=args.lrate)
        sys.stdout.write(
            '# SET Learning %s: initial learning rate: %e\n' %
            (args.optimizer, optimizer.alpha))
    elif args.optimizer == 'MomentumSGD':
        optimizer = chaOpt.MomentumSGD(lr=args.lrate)
        sys.stdout.write(
            '# SET Learning %s: initial learning rate: %e\n' %
            (args.optimizer, optimizer.lr))
    elif args.optimizer == 'AdaDelta':
        optimizer = chaOpt.AdaDelta(rho=args.lrate)
        sys.stdout.write(
            '# SET Learning %s: initial learning rate: %e\n' %
            (args.optimizer, optimizer.rho))
    else:
        assert 0, "ERROR"

    optimizer.setup(EncDecAtt.model)  # ここでoptimizerにモデルを貼り付け
    if args.optimizer == 'Adam':
        optimizer.t = 1  # warning回避のちょっとしたhack 本来はするべきではない

    return optimizer


def setGradClip(args, optimizer):
    clipV = args.grad_clip
    sys.stdout.write('# USE gradient clipping: %f\n' % (clipV))
    clip_obj = Chainer_GradientClipping_rmk_v1(clipV)
    optimizer.add_hook(clip_obj)

    return clip_obj


#########################################
# 学習用の関数本体
def train_model(args):

    prepD = None
    if args.setting_file:
        sys.stdout.write(
            '# Loading initial data  config=[%s] model=[%s] \n' %
            (args.setting_file, args.init_model_file))
        EncDecAtt = pickle.load(open(args.setting_file, 'rb'))
        prepD = PrepareData(EncDecAtt)
    else:
        prepD = PrepareData(args)
        sys.stdout.write('# encVocab [%s] \n' % (args.encVocabFile))
        encoderVocab = prepD.readVocab(args.encVocabFile)
        # decoderVocab = prepD.readVocab(args.decVocabFile)
        decoderVocab = []
        for i, fname in enumerate(args.decVocabFile.split(':')):
            sys.stdout.write(
                '# decVocab [%s] [%s] \n' %
                (args.decVocabFile, fname))
            decoderVocab.append(prepD.readVocab(fname))
        #########
        EncDecAtt = EncoderDecoderAttention(encoderVocab, decoderVocab, args)

    if args.outputFile:
        outputFile = open(args.outputFile + '.setting', 'wb')
        pickle.dump(EncDecAtt, outputFile)
        outputFile.close()
    ########################################
    EncDecAtt.initModel()  # ここでモデルをいったん初期化
    args.eDim = EncDecAtt.eDim  # 念の為，強制置き換え
    args.hDim = EncDecAtt.hDim  # 念の為，強制置き換え

    sys.stdout.write('#####################\n')
    sys.stdout.write('# [Params] {}'.format(args))
    sys.stdout.write('#####################\n')

    EncDecAtt.setToGPUs(args)  # ここでモデルをGPUに貼り付ける

    optimizer = setOptimizer(args, EncDecAtt)
    clip_obj = setGradClip(args, optimizer)

    ########################################
    # 学習済みの初期モデルがあればをここで読み込む
    if args.setting_file and args.init_model_file:
        sys.stderr.write('Load model from: [%s]\n' % (args.init_model_file))
        chaSerial.load_npz(args.init_model_file, EncDecAtt.model)
    else:  # 学習済みの初期モデルがなければパラメタを全初期化する
        EncDecAtt.setInitAllParameters(optimizer, init_type=args.init_type,
                                       init_scale=args.init_scale)
        if args.init_emb_by_w2v_file:
            EncDecAtt.readWord2vecResult(args.init_emb_by_w2v_file,
                                         args.init_emb_by_w2v_mode)
            sys.stdout.write("############ Current Parameters BEGIN\n")
            EncDecAtt.printAllParameters(optimizer)
            sys.stdout.write("############ Current Parameters END\n")

    ########################################
    # ここでencoder側/decoder側のデータを全て読み込む
    if True:
        sys.stdout.write(
            '# encTrainDataFile [{}] vocab=[{}]\n'
            .format(args.encDataFile, len(EncDecAtt.encoderVocab)))
        encSentLenDict = prepD.makeSentenceLenDict(
            args.encDataFile, EncDecAtt.encoderVocab, input_side=True)
        # dec側は複数のファイルがあることを仮定(一つでもよい)
        sys.stdout.write(
            '# decTrainDataFile [{}]\n'.format(args.decDataFile))
        decDataFile = []
        for z, fname in enumerate(args.decDataFile.split(':')):
            sys.stdout.write(
                '# decTrainDataFile [{}] [{}] vocab=[{}]\n'
                .format(z, fname, len(EncDecAtt.decoderVocab[z])))
            decDataFile.append(fname)
        assert len(decDataFile) == len(EncDecAtt.decoderVocab)
        decSentLenDict = [0] * len(decDataFile)
        for z, fname in enumerate(decDataFile):
            decSentLenDict[z] = prepD.makeSentenceLenDict(
                fname, EncDecAtt.decoderVocab[z], input_side=False)
        ##
        # decSentLenDict = prepD.makeSentenceLenDict(
        #     args.decDataFile, EncDecAtt.decoderVocab, input_side=False)
        #####
        if args.mode_data_shuffle == 0:  # default
            trainData = prepD.makeBatch4Train(
                encSentLenDict,
                decSentLenDict,
                args.batch_size,
                shuffle_flag=True)
    if args.encDevelDataFile and args.decDevelDataFile:
        sys.stdout.write(
            '# encDevelDataFile [{}] vocab=[{}]\n'
            .format(args.encDevelDataFile, len(EncDecAtt.encoderVocab)))
        encSentLenDictDevel = prepD.makeSentenceLenDict(
            args.encDevelDataFile, EncDecAtt.encoderVocab, input_side=True)
        # dec側は複数のファイルがあることを仮定(一つでもよい)
        sys.stdout.write(
            '# decDevelDataFile [{}]\n'.format(args.decDevelDataFile))
        decDevelDataFile = []
        for z, fname in enumerate(args.decDevelDataFile.split(':')):
            sys.stdout.write(
                '# decDevelDataFile [{}] [{}] vocab=[{}]\n'
                .format(z, fname, len(EncDecAtt.decoderVocab[z])))
            decDevelDataFile.append(fname)
        assert len(decDevelDataFile) == len(EncDecAtt.decoderVocab)
        decSentLenDictDevel = [0] * len(decDevelDataFile)
        for z, fname in enumerate(decDevelDataFile):
            decSentLenDictDevel[z] = prepD.makeSentenceLenDict(
                fname, EncDecAtt.decoderVocab[z], input_side=False)
        ##
        # decSentLenDictDevel = prepD.makeSentenceLenDict(
        #     args.decDevelDataFile, EncDecAtt.decoderVocab, input_side=False)
        ######
        develData = prepD.makeBatch4Train(
            encSentLenDictDevel, decSentLenDictDevel, args.batch_size,
            shuffle_flag=False)

    prevLossDevel = 1.0e+100
    prevAccDevel = 0
    prevLossTrain = 1.0e+100
    # 学習のループ
    for epoch in six.moves.range(args.epoch + 1):
        ####################################
        # devの評価モード
        if args.encDevelDataFile and args.decDevelDataFile:
            train_mode = 0
            # dropout_rate = 0
            chainer.global_config.train = False
            chainer.global_config.enable_backprop = False
            begin = time.time()
            sys.stdout.write(
                ('# Dev. data | total mini batch bucket size = {0}\n'.format(
                    len(develData))))
            tInfo = train_model_sub(train_mode, epoch, develData, EncDecAtt,
                                    None, clip_obj, begin, args)
            msgA = tInfo.print_strings(train_mode, epoch, 0, 0, 0, begin, args)
            dL = prevLossDevel - float(tInfo.lossVal)
            sys.stdout.write('\r# Dev.Data | %s | diff: %e\n' % (
                msgA, dL / max(1, tInfo.instanceNum)))
            # learning rateを変更するならここ
            if args.optimizer == 'SGD':
                if epoch >= args.lrate_decay_at or (
                        epoch >= args.lrate_no_decay_to and
                        tInfo.lossVal > prevLossDevel and
                        tInfo.corTot < prevAccDevel):
                    optimizer.lr = max(
                        args.lrate * 0.01, optimizer.lr * args.lrate_decay)
                sys.stdout.write('SGD Learning Rate: %s  (initial: %s)\n' % (
                    optimizer.lr, args.lrate))
            elif args.optimizer == 'Adam':
                if epoch >= args.lrate_decay_at or (
                        epoch >= args.lrate_no_decay_to and
                        tInfo.lossVal > prevLossDevel and
                        tInfo.corTot < prevAccDevel):
                    optimizer.alpha = max(
                        args.lrate * 0.01, optimizer.alpha * args.lrate_decay)
                sys.stdout.write(
                    ('Adam Learning Rate: t=%s lr=%s ep=%s '
                     'alpha=%s beta1=%s beta2=%s\n' % (
                         optimizer.t, optimizer.lr, optimizer.epoch,
                         optimizer.alpha, optimizer.beta1, optimizer.beta2)))
            # develのlossとaccを保存
            prevLossDevel = tInfo.lossVal
            prevAccDevel = tInfo.corTot
        ####################################
        # 学習モード
        # shuffleしながらmini batchを全て作成する
        # epoch==0のときは長い順（メモリ足りない場合の対策 やらなくてもよい）
        if epoch < args.epoch:  # 学習は必ず行うことが前提
            train_mode = 1
            # dropout_rate = args.dropout_rate
            chainer.global_config.train = True
            chainer.global_config.enable_backprop = True
            begin = time.time()
            if args.mode_data_shuffle == 0:  # default
                # encLenの長さでまとめたものをシャッフルする
                random.shuffle(trainData)
            elif args.mode_data_shuffle == 1:  # minibatchも含めてshuffle
                trainData = prepD.makeBatch4Train(
                    encSentLenDict, decSentLenDict, args.batch_size, True)
            # minibatchも含めてshuffle + 最初のiterationは長さ順 (debug用途)
            elif args.mode_data_shuffle == 2:
                trainData = prepD.makeBatch4Train(
                    encSentLenDict, decSentLenDict,
                    args.batch_size, (epoch != 0))
            else:
                assert 0, "ERROR"
            sys.stdout.write(
                ('# Train | data shuffle | total mini batch bucket size = {0} '
                 '| Time: {1:10.4f}\n'.format(
                     len(trainData), time.time() - begin)))
            # sys.stderr.write(
            #   ('# TRAIN epoch {} drop rate={} | CHAINER CONFIG  [{}] \n'
            #    .format(epoch, dropout_rate, chainer.global_config.__dict__)))
            # 学習の実体
            begin = time.time()
            tInfo = train_model_sub(train_mode, epoch, trainData, EncDecAtt,
                                    optimizer, clip_obj, begin, args)
            msgA = tInfo.print_strings(train_mode, epoch, 0, 0, 0, begin, args)
            dL = prevLossTrain - float(tInfo.lossVal)
            sys.stdout.write('\r# Train END %s | diff: %e\n' % (
                msgA, dL / max(1, tInfo.instanceNum)))
            prevLossTrain = tInfo.lossVal
            ####################################
            # モデルの保存
            if (args.outputFile and
                (epoch + 1 == args.epoch or
                 (args.outEach != 0 and (epoch + 1) % args.outEach == 0))):
                try:
                    outputFileName = args.outputFile + '.epoch%s' % (epoch + 1)
                    sys.stdout.write(
                        "#output model [{}]\n".format(outputFileName))
                    chaSerial.save_npz(
                        outputFileName, copy.deepcopy(
                            EncDecAtt.model).to_cpu(), compression=True)
                    # chaSerial.save_hdf5(
                    #    outputFileName, copy.deepcopy(
                    #        EncDecAtt.model).to_cpu(), compression=9)
                except Exception as e:
                    # メモリエラーなどが発生しても処理を終了せずに
                    # そのサンプルをスキップして次に進める
                    sys.stdout.write(
                        '\r# SAVE Error? Skip! {} | {}\n'.format(
                            outputFileName, type(e)))
                    sys.stdout.flush()
    ####################################
    sys.stdout.write('Done\n')
####################################


####################################
# 以下，評価時だけ使う関数
def updateBeamThreshold__2(queue, input, max_queue_size=-1):
    # list内の要素はlist,タプル，かつ，0番目の要素はスコアを仮定
    queue_size = len(queue)
    # sys.stderr.write('### QUEUE {} {} | {} {}\n'.format(
    #    len(queue), max_queue_size,  queue[-1][0], input[0]))
    if len(queue) == 0:
        queue.append(input)
    elif max_queue_size > 0 and queue_size < max_queue_size:
        for i in six.moves.range(queue_size):
            if queue[i][0] <= input[0]:
                continue
            tmp = queue[i]
            queue[i] = input
            input = tmp
        queue.append(input)
        # sys.stderr.write('### QUEUE APPEND {} {} | {} {}\n'.format(
        #    len(queue), max_queue_size,  queue[-1][0], input[0]))
    else:
        # TODO 線形探索なのは面倒なので 効率を上げるためには要修正
        for i in six.moves.range(queue_size):
            if queue[i][0] <= input[0]:
                continue
            tmp = queue[i]
            queue[i] = input
            input = tmp
    return queue


def decodeByBeamFast(
        EncDecAttModels,
        encSent,
        max_length,
        beam_size,
        args,
        fil_a,
        fil_b,
        fil_c):
    train_mode = 0  # 評価なので
    encInfo = [0] * len(EncDecAttModels)
    aList = [0] * len(EncDecAttModels)
    prevOutIn = [0] * len(EncDecAttModels)

    for i, EncDecAtt in enumerate(EncDecAttModels):
        encInfo[i] = EncDecAtt.encodeSentenceFWD(
            train_mode, encSent, args, 0.0)
        if args.gpu_enc != args.gpu_dec:  # encとdecが別GPUの場合
            chainer.cuda.get_device(args.gpu_dec).use()  # dec側のgpuで処理をするため
        aList[i], prevOutIn[i] = EncDecAtt.prepareDecoder(encInfo[i])
    encLen = encInfo[0].encLen

    idx_bos = EncDecAttModels[0].decoderVocab[0]['<s>']
    idx_eos = EncDecAttModels[0].decoderVocab[0]['</s>']
    idx_unk = EncDecAttModels[0].decoderVocab[0]['<unk>']

    xp = cuda.get_array_module(encInfo[0].lstmVars[0].data)
    if args.wo_rep_w:
        WFilter = xp.zeros(
            (1, EncDecAttModels[0].decVocabSize[0]), dtype=xp.float32)
    else:
        WFilter = None

    type_counter = xp.zeros((1, 2), dtype=xp.int32)
    type_counter[0][1] = encLen

    beam = [(0, [idx_bos], idx_bos, [x.lstmVars for x in encInfo],
             prevOutIn, WFilter, type_counter)]
    dummy_b = (1.0e+999, [idx_bos], idx_bos, None, None, WFilter, type_counter)

    for i in six.moves.range(max_length + 1):  # for </s>
        # newBeam = [dummy_b]*beam_size
        newBeam = [dummy_b] * 1  # for small output size

        cMBSize = len(beam)

        nextWordProb_a = xp.zeros(
            (cMBSize, EncDecAttModels[0].decVocabSize[0]), dtype=xp.float32)
        epsilon_a = xp.ones(
            (cMBSize,
             EncDecAttModels[0].decVocabSize[0]),
            dtype=xp.float32) * 0.000001
        lstm_states_a_list = [0] * len(EncDecAttModels)
        next_h4_a_list = [0] * len(EncDecAttModels)
        for j, EncDecAtt in enumerate(EncDecAttModels):
            ###################################################################
            # beamで分割されているものを一括処理するために miniBatchとみなして処理 準備としてbeamの情報を結合
            # beam内の候補をminibatchとして扱うために，axis=0 を 1から cMBSizeに拡張するためにbroadcast
            biH0 = chaFunc.broadcast_to(
                encInfo[j].attnList, (cMBSize, encLen, EncDecAtt.hDim))
            if EncDecAtt.attn_mode == 1:
                aList_a = biH0
            elif EncDecAtt.attn_mode == 2:
                t = chaFunc.broadcast_to(
                    chaFunc.reshape(
                        aList[j], (1, encLen, EncDecAtt.hDim)),
                    (cMBSize, encLen, EncDecAtt.hDim))
                aList_a = chaFunc.reshape(
                    t, (cMBSize * encLen, EncDecAtt.hDim))
                # TODO: 効率が悪いのでencoder側に移動したい
            else:
                assert 0, "ERROR"

            # axis=1 (defaultなので不要)= hstack
            lstm_states_a = chaFunc.concat([x[3][j] for x in beam])
            # concat(a, axis=0) == vstack(a)
            prevOutIn_a = chaFunc.concat([x[4][j] for x in beam], axis=0)
            # decoder側の単語を取得
            wordIndex = np.array([x[2] for x in beam],
                                 dtype=np.int32)  # 一つ前の予測結果から単語を取得
            inputEmbList = EncDecAtt.model.decOutputL.getInputEmbeddings(
                wordIndex, 0, args)
            ###################################################################
            ##
            hOut, lstm_states_a = EncDecAtt.processDecLSTMOneStep(
                inputEmbList, lstm_states_a, prevOutIn_a, args, 0.0)
            next_h4_a = EncDecAtt.calcAttention(hOut, biH0, aList_a,
                                                encLen, cMBSize, args)
            lstm_states_a_list[j] = lstm_states_a
            next_h4_a_list[j] = next_h4_a
            nextWordProb_a += EncDecAtt.model.decOutputL.getProbSingle2(
                0, [next_h4_a], cMBSize, args)  # TODO

            # oVector_a = EncDecAtt.generateWord(next_h4_a_list[j],
            #                                    encLen, cMBSize, args, 0.0)
        ###################
        nextWordProb_a /= len(EncDecAttModels)
        nextWordProb_a = - chaFunc.clip(
            chaFunc.log(nextWordProb_a + epsilon_a), -1.0e+99, 0.0).data
        # ここでlogにしているので，これ以降は nextWordProb_a
        # は大きい値ほど選択されないと言う意味

        ######
        if args.wo_rep_w:
            WFilter_a = xp.concatenate([x[5] for x in beam], axis=0)
            nextWordProb_a += WFilter_a
        # 絶対に出てほしくない出力を強制的に選択できないようにするために大きな値をセットする
        if args.use_bos:  # BOSは出さない設定の場合
            if i != 0:  # EOSは文の先頭だけ許可（逆に言うと，文の先頭以外は不許可）
                nextWordProb_a[:, idx_bos] = 1.0e+100
        else:
            nextWordProb_a[:, idx_bos] = 1.0e+100  # BOS
        if args.wo_unk:  # UNKは出さない設定の場合
            nextWordProb_a[:, idx_unk] = 1.0e+100

        #############################
        if args.use_restrict_decoding:
            for z, b in enumerate(beam):
                t_type_counter = b[6]
                if t_type_counter[0][1] == 0:  # XXを使い切った => XXと開き括弧の禁止
                    nextWordProb_a[z] = nextWordProb_a[z] + \
                        (fil_a + fil_b) * 1.0e+100  # 1.0e+100
                else:
                    nextWordProb_a[z, idx_eos] = 1.0e+100
                if t_type_counter[0][0] == 0:  # 開き閉じ括弧の数が同じ => 閉じ括弧の禁止
                    nextWordProb_a[z] = nextWordProb_a[z] + \
                        fil_c * 1.0e+100  # 1.0e+100
                else:
                    nextWordProb_a[z, idx_eos] = 1.0e+100
        #############################

        #######################################################################
        # beam_size個だけ使う，使いたくない要素は上の値変更処理で事前に省く
        if args.gpu_enc >= 0:
            nextWordProb_a = nextWordProb_a.get()  # sort のためにCPU側に移動
        if beam_size >= nextWordProb_a.shape[-1]:  # argpartitionは最後の要素をならべかえる
            sortedIndex_a = bn.argpartition(
                nextWordProb_a, nextWordProb_a.shape[-1] - 1)
        else:
            sortedIndex_a = bn.argpartition(
                nextWordProb_a, beam_size)[:, :beam_size]
        #######################################################################

        for z, b in enumerate(beam):
            # まず，EOSまで既に到達している場合はなにもしなくてよい (beamはソートされていることが条件)
            if b[2] == idx_eos:
                newBeam = updateBeamThreshold__2(newBeam, b, beam_size + 1)
                continue
            ##
            flag_force_eval = False
            if i == max_length:  # mode==0,1,2: free,word,char
                flag_force_eval = True

            if not flag_force_eval and b[0] > newBeam[-1][0]:
                continue
            # 3
            # 次のbeamを作るために準備
            lstm_states = [x[:, z:z + 1, ] for x in lstm_states_a_list]
            next_h4 = [x[z:z + 1, ] for x in next_h4_a_list]
            nextWordProb = nextWordProb_a[z]
            ###################################
            # 長さ制約的にEOSを選ばなくてはいけないという場合
            if flag_force_eval:
                wordIndex = idx_eos
                newProb = nextWordProb[wordIndex] + b[0]
                if args.wo_rep_w:
                    tWFilter = b[5].copy()
                    tWFilter[:, wordIndex] += 1.0e+100
                else:
                    tWFilter = b[5]
                t_type_counter = b[6].copy()
                nb = (
                    newProb,
                    b[1][:] +
                    [wordIndex],
                    wordIndex,
                    lstm_states,
                    next_h4,
                    tWFilter,
                    t_type_counter)
                newBeam = updateBeamThreshold__2(newBeam, nb, beam_size + 1)
                continue
            # ここまでたどり着いたら最大beam個評価する
            # 基本的に sortedIndex_a[z] は len(beam) 個しかない
            for wordIndex in sortedIndex_a[z]:
                newProb = nextWordProb[wordIndex] + b[0]
                if newProb > newBeam[-1][0]:
                    continue
                    # break
                # ここまでたどり着いたら入れる
                if args.wo_rep_w:
                    tWFilter = b[5].copy()
                    tWFilter[:, wordIndex] += 1.0e+100
                else:
                    tWFilter = b[5]

                #########################
                t_type_counter = b[6].copy()
                if args.use_restrict_decoding:
                    if wordIndex != 0:
                        tw = EncDecAttModels[0].index2decoderWord[0][wordIndex]
                    else:
                        tw = "_XX_"
                    if tw == ',' or tw == '.' or tw == '``' or \
                        tw == '\'\'' or tw == ':' or tw == 'XX' or tw[
                            0] == '_':
                        t_type_counter[0][1] -= 1
                    elif tw[0] == '(':
                        t_type_counter[0][0] += 1
                    elif tw[0] == ')':
                        t_type_counter[0][0] -= 1
                    else:
                        pass
                #########################
                nb = (
                    newProb,
                    b[1][:] +
                    [wordIndex],
                    wordIndex,
                    lstm_states,
                    next_h4,
                    tWFilter,
                    t_type_counter)
                newBeam = updateBeamThreshold__2(newBeam, nb, beam_size + 1)
                #####
        ################
        # 一時刻分の処理が終わったら，入れ替える
        beam = newBeam[0:-1]  # the last element can be dummy
        # beam = newBeam
        if all([True if b[2] == idx_eos else False for b in beam]):
            break
        # 次の入力へ
    # for PTBParsing
    beam = [(b[0], b[1], b[3], b[6], [EncDecAttModels[0].index2decoderWord[0][
             z] if z != 0 else "XX" for z in b[1]]) for b in beam]

    return beam


def rerankingByLengthNormalizedLoss(beam, wposi):
    beam.sort(key=lambda b: b[0] / (len(b[wposi]) - 1))
    return beam


def decodeByBeamFast2OneBest(
        EncDecAttModels,
        encSent,
        decMaxLen,
        beam_size,
        args,
        fil_a,
        fil_b,
        fil_c):
    outputBeam = decodeByBeamFast(
        EncDecAttModels,
        encSent,
        decMaxLen,
        beam_size,
        args,
        fil_a,
        fil_b,
        fil_c)
    wposi = 4
    # outloop = 1

    # 長さに基づく正規化 このオプションを使うことを推奨
    if args.length_normalized:
        outputBeam = rerankingByLengthNormalizedLoss(outputBeam, wposi)

    if args.output_all_beam > 0:
        # outloop = args.beam_size
        sys.stdout.write('{}\n'.format(len(outputBeam)))

    outputList = outputBeam[0][wposi]
    if outputList[-1] != '</s>':
        outputList.append('</s>')
    out_text = ' '.join(outputList[1:len(outputList) - 1])
    return out_text


# テスト部本体
def ttest_model(args):
    # init_modelがスペース区切りで複数与えられているときは、ensembleする
    model_files = args.init_model_file.split(':')
    EncDecAttModels = [0] * len(model_files)

    for i, model_file in enumerate(model_files):
        EncDecAtt = pickle.load(open(args.setting_file, 'rb'))
        EncDecAtt.initModel()
        if args.setting_file and model_file:  # モデルをここで読み込む
            sys.stderr.write('Load model from: [%s]\n' % (model_file))
            chaSerial.load_npz(model_file, EncDecAtt.model)
        else:
            assert 0, "ERROR"
        EncDecAtt.setToGPUs(args)
        EncDecAttModels[i] = EncDecAtt
    prepD = PrepareData(EncDecAttModels[0])
    EncDecAtt = None  # 念のため

    sys.stderr.write('Finished loading model\n')

    sys.stderr.write('max_length is [%d]\n' % args.max_length)
    sys.stderr.write('w/o generating unk token [%r]\n' % args.wo_unk)
    sys.stderr.write('w/o generating the same words in twice [%r]\n' %
                     args.wo_rep_w)
    sys.stderr.write('beam size is [%d]\n' % args.beam_size)
    sys.stderr.write('output is [%s]\n' % args.outputFile)

    ####################################
    decMaxLen = args.max_length

    ####################################
    fil_a = xp.zeros(
        (len(
            EncDecAttModels[0].index2decoderWord[0],
        )),
        dtype=xp.int32)
    fil_b = xp.zeros(
        (len(
            EncDecAttModels[0].index2decoderWord[0],
        )),
        dtype=xp.int32)
    fil_c = xp.zeros(
        (len(
            EncDecAttModels[0].index2decoderWord[0],
        )),
        dtype=xp.int32)

    ####################################
    if args.use_restrict_decoding:
        sys.stderr.write(
            '### vocab {} | {}\n'.format(
                type(
                    EncDecAttModels[0].index2decoderWord[0]),
                EncDecAttModels[0].index2decoderWord[0]))
        for i, tw in EncDecAttModels[0].index2decoderWord[0].items():
            sys.stderr.write('### vocab {} => {}\n'.format(i, tw))
            if tw == ',' or tw == '.' or tw == '``' or \
               tw == '\'\'' or tw == ':' or tw == 'XX' or tw[0] == '_':
                fil_a[i] = 1
            elif tw[0] == '(':
                fil_b[i] = 1
            elif tw[0] == ')':
                fil_c[i] = 1
            else:
                pass
        sys.stderr.write('### fil_a {} {}\n'.format(fil_a.shape, fil_a))
        sys.stderr.write('### fil_b {} {}\n'.format(fil_b.shape, fil_b))
        sys.stderr.write('### fil_c {} {}\n'.format(fil_c.shape, fil_c))
        sys.stderr.write('### total {}\n'.format(fil_a + fil_b + fil_c))
    ####################################

    begin = time.time()
    counter = 0
    # TODO: codecsでないとエラーが出る環境がある？ 要調査 不要ならioにしたい
    with io.open(args.encDataFile, encoding='utf-8') as f:
        # with codecs.open(args.encDataFile, encoding='utf-8') as f:
        for sentence in f:
            sentence = sentence.strip()  # stripを忘れずに．．．
            # ここでは，入力された順番で一文ずつ処理する方式のみをサポート
            sourceSentence = prepD.inputsentence2index(
                sentence, EncDecAttModels[0].encoderVocab, input_side=True)
            sourceSentence = [sourceSentence, ]  # minibatch化と同じ意味
            # 1文ずつ処理するので，test時は基本必ずminibatch=1になる
            outputBeam = decodeByBeamFast(
                EncDecAttModels,
                sourceSentence,
                decMaxLen,
                args.beam_size,
                args,
                fil_a,
                fil_b,
                fil_c)
            wposi = 4
            outloop = 1

            # 長さに基づく正規化 このオプションを使うことを推奨
            if args.length_normalized:
                outputBeam = rerankingByLengthNormalizedLoss(outputBeam, wposi)

            if args.output_all_beam > 0:
                outloop = args.beam_size
                sys.stdout.write('{}\n'.format(len(outputBeam)))

            for i in six.moves.range(outloop):
                if i >= len(outputBeam):
                    break
                outputList = outputBeam[i][wposi]
                # score = outputBeam[i][0]
                if outputList[-1] != '</s>':
                    outputList.append('</s>')
                # if args.outputAllBeam > 0:
                # sys.stdout.write("# {} {} {}\n".format(i, score,
                # len(outputList)))

                sys.stdout.write('{}\n'.format(
                    ' '.join(outputList[1:len(outputList) - 1])))
                # charlenList = sum([ len(z)+1 for z in
                # 文末の空白はカウントしないので-1
                # outputList[1:len(outputList) - 1] ])-1
            counter += 1
            sys.stderr.write('\nSent.Num: %5d %s | cand=%d | Time: %10.4f ' %
                             (counter, outputBeam[0][wposi], len(outputBeam),
                              time.time() - begin))
            if args.use_restrict_decoding:
                if outputBeam[0][3][0][0] != 0 or outputBeam[0][3][0][1] != 0:
                    sys.stderr.write(
                        ' ### WARNING ### ||| {} '.format(
                            outputBeam[0][3]))
    sys.stderr.write('\nDONE: %5d | Time: %10.4f\n' %
                     (counter, time.time() - begin))


def parse_options_encdecattn():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu-enc',
        dest='gpu_enc',
        default=-1,
        type=int,
        help='GPU ID for encoder (negative value indicates CPU)')
    parser.add_argument(
        '--gpu-dec',
        dest='gpu_dec',
        default=-1,
        type=int,
        help=('GPU ID for decoder including attention part '
              '(negative value indicates CPU)'))
    parser.add_argument(
        '-T',
        '--train-test-mode',
        dest='train_mode',
        default='train',
        help=('select train or test mode [string] '
              'default=train option=train, test'))
    parser.add_argument(
        '-V',
        '--verbose',
        dest='verbose',
        default=1,
        type=int,
        help='verbose level [int] default=1, option=0, 1, 2')
    parser.add_argument(
        '-D',
        '--embed-dim',
        dest='eDim',
        default=512,
        type=int,
        help=('dimensions of embedding layers in both encoder/decoder '
              '[int] default=512'))
    parser.add_argument(
        '-H',
        '--hidden-dim',
        dest='hDim',
        default=512,
        type=int,
        help='dimensions of all hidden layers [int] default=512')
    parser.add_argument(
        '-N',
        '--num-rnn-layers',
        dest='n_layers',
        default=2,
        type=int,
        help=('number of RNN (LSTM) layers in both encoder/decoder '
              '[int] default=2'))
    parser.add_argument(
        '-E',
        '--epoch',
        dest='epoch',
        default=13,
        type=int,
        help='number of epoch [int] default=13')
    parser.add_argument(
        '-B',
        '--batch-size',
        dest='batch_size',
        default=128,
        type=int,
        help='mini batch size [int] default=128')
    parser.add_argument(
        '-O',
        '--output',
        dest='outputFile',
        default='',
        help=('name of output (model) file [string] '
              'default=(No output)'))
    parser.add_argument(
        '--out-each',
        dest='outEach',
        default=0,
        type=int,
        help='output file by each epoch')

    parser.add_argument(
        '--enc-vocab-file',
        dest='encVocabFile',
        default='',
        help='filename of encoder (input)-side vocabulary')
    parser.add_argument(
        '--dec-vocab-file',
        dest='decVocabFile',
        default='',
        help='filename of decoder (output)-side vocabulary')
    parser.add_argument(
        '--enc-data-file',
        dest='encDataFile',
        default='',
        help='filename of encoder (input)-side data for training')
    parser.add_argument(
        '--dec-data-file',
        dest='decDataFile',
        default='',
        help='filename of decoder (output)-side data for trainig')
    parser.add_argument(
        '--enc-devel-data-file',
        dest='encDevelDataFile',
        default='',
        help='filename of encoder (input)-side data for development data')
    parser.add_argument(
        '--dec-devel-data-file',
        dest='decDevelDataFile',
        default='',
        help='filename of decoder (output)-side data for development data')

    parser.add_argument(
        '--lrate',
        dest='lrate',
        default=1.0,
        type=float,
        help='learning rate [float] default=1.0')
    parser.add_argument(
        '--lrate-no-decay-to',
        dest='lrate_no_decay_to',
        default=9,
        type=int,
        help='start decay after this epoch [int] default=9')
    parser.add_argument(
        '--lrate-decay-at',
        dest='lrate_decay_at',
        default=9,
        type=int,
        help='start decay after this epoch [int] default=9')
    parser.add_argument(
        '--lrate-decay',
        dest='lrate_decay',
        default=0.5,
        type=float,
        help='decay learning rate [float] default=0.5')

    parser.add_argument(
        '--optimizer',
        dest='optimizer',
        default='SGD',
        help='optimizer type [string] default=SGD, option: MomentumSGD, Adam')
    parser.add_argument(
        '--gradient-clipping',
        dest='grad_clip',
        default=5.0,
        type=float,
        help='gradient clipping threshold [float] default=5.0')
    parser.add_argument(
        '--dropout-rate',
        dest='dropout_rate',
        default=0.3,
        type=float,
        help='dropout rate [float] default=0.3')
    parser.add_argument(
        '--embeddings-always-cpu',
        dest='flag_emb_cpu',
        default=False,
        action='store_true',
        help=('embeddings are alwasy stored on cpu regardless of GPU usage '
              '[bool] default=False'))
    parser.add_argument(
        '-M',
        '--init-model',
        dest='init_model_file',
        default='',
        help='filename of model file')
    parser.add_argument(
        '-S',
        '--setting',
        dest='setting_file',
        default='',
        help='filename of setting file')
    parser.add_argument(
        '--initializer-scale',
        dest='init_scale',
        default=0.1,
        type=float,
        help='scaling factor for random initializer [float] default=0.1')
    parser.add_argument(
        '--initializer-type',
        dest='init_type',
        default="uniform",
        help=('select initializer [string] default=uniform '
              'option=chainer_default, normal'))

    # 速度を稼ぎたいときはOFF(0)にする
    parser.add_argument(
        '--eval-accuracy',
        dest='doEvalAcc',
        default=0,
        type=int,
        help='with/without evaluating accuracy during training')

    # modelのやhidden layerの次元数の変更を伴うオプション
    parser.add_argument(
        '--use-encoder-bos-eos',
        dest='flag_enc_boseos',
        default=0,
        type=int,
        help=('with/without adding BOS and EOS for encoder '
              '(input-side) sentences [int] default=0\n  '
              'NOTE: this option is basically imported from '
              'the "-start_symbol" option in the seq2seq-attn tool'))
    parser.add_argument(
        '--merge-encoder-fwbw',
        dest='flag_merge_encfwbw',
        default=0,
        type=int,
        help=('how to calculate bidirectional (fw/bk) LSTM in encoder [int] '
              'default=0 (separate) option=1 (merge)'))
    parser.add_argument(
        '--attention-mode',
        dest='attn_mode',
        default=1,
        type=int,
        help=('attention mode [int] default=1 (bilinear), '
              'option=0 (w/o attention), 2 (MLP) '))
    parser.add_argument(
        '--use-decoder-inputfeed',
        dest='flag_dec_ifeed',
        default=1,
        type=int,
        help=('w/ or w/o using previous final hidden states of '
              'next target input'))

    parser.add_argument(
        '--shuffle-data-mode',
        dest='mode_data_shuffle',
        default=0,
        type=int,
        help=('shuffle data mode [int] default=0 '
              '(only minibatch bucket shuffle) option=1 (shuffle all data)'))

    # decoder options for test
    parser.add_argument(
        '--max-length',
        dest='max_length',
        default=200,
        type=int,
        help=('[decoder option] '
              'the maximum number of words in output'))
    parser.add_argument(
        '--use-bos',
        dest='use_bos',
        default=False,
        action='store_true',
        help=('[decoder option] '
              'with or without using UNK tokens in output [bool] '
              'default=False'))
    parser.add_argument(
        '--without-unk',
        dest='wo_unk',
        default=False,
        action='store_true',
        help=('[decoder option] '
              'with or without using UNK tokens in output [bool] '
              'default=False'))
    parser.add_argument(
        '--beam-size',
        dest='beam_size',
        default=1,
        type=int,
        help=('[decoder option] '
              'beam size in beam search decoding [int] default=1'))
    parser.add_argument(
        '--without-repeat-words',
        dest='wo_rep_w',
        default=False,
        action='store_true',
        help=('[decoder option] '
              'restrict each word to appear at most once [bool] '
              'default=False'))
    parser.add_argument(
        '--length-normalized',
        dest='length_normalized',
        default=False,
        action='store_true',
        help=('normalize the scores by the sentence length and reranking '
              '[bool] default=False'))

    parser.add_argument(
        '--random-seed',
        dest='seed',
        default=2723,
        type=int,
        help='random seed [int] default=2723')

    parser.add_argument(
        '--output-layer-type',
        dest='output_layer_type',
        default=0,
        type=int,
        help=('shuffle data mode [int] default=0 '
              '(only minibatch bucket shuffle) option=1 (shuffle all data)'))

    parser.add_argument(
        '--init-emb-by-w2v-file',
        dest='init_emb_by_w2v_file',
        default='',
        help='specify the name of file representing vector for initialization')

    parser.add_argument(
        '--init-emb-by-w2v-mode',
        dest='init_emb_by_w2v_mode',
        default=0,
        type=int,
        help='specify the mode of representing vector for initialization')

    parser.add_argument(
        '--output-all-beam',
        dest='output_all_beam',
        default=0,
        type=int,
        help='output all candidates in beam')

    parser.add_argument(
        '--dec-emb-tying',
        dest='decEmbTying',
        default=False,
        action='store_true',
        help=(
            'share parameters between decoder embedding '
            'and output layer [bool] '
            'default=False'))

    parser.add_argument(
        '--use-restrict-decoding',
        dest='use_restrict_decoding',
        default=False,
        action='store_true',
        help=('restrict decoding [bool] '
              'default=False'))

    args = parser.parse_args()
    return args


#######################################
# ## main関数
if __name__ == "__main__":

    ##################################################################
    # コマンドラインオプション取得
    args = parse_options_encdecattn()

    # ここから開始
    sys.stderr.write('CHAINER VERSION [{}] \n'.format(chainer.__version__))
    # python 3.5以降 ＋ chainer version2以降のみサポート
    args.chainer_version_check = [int(z)
                                  for z in chainer.__version__.split('.')[:2]]
    sys.stderr.write('CHAINER VERSION check [{}]\n'.format(
        args.chainer_version_check))
    if args.chainer_version_check[0] < 2 or sys.version_info < (3, 5):
        assert 0, "ERROR: not supported version for this code"
    # sys.stderr.write(
    #     'CHAINER CONFIG  [{}] \n'.format(chainer.global_config.__dict__))
    # プライマリのGPUをセット 或いはGPUを使わない設定にする
    if args.gpu_enc >= 0 and args.gpu_dec >= 0:
        import cupy as xp
        cuda.check_cuda_available()
        cuda.get_device(args.gpu_enc).use()
        sys.stderr.write('CUPY VERSION [{}]\n'.format(xp.__version__))
        # cudnn = cuda.cudnn
        # libcudnn = cuda.cudnn.cudnn
        sys.stderr.write(
            'w/  using GPU [%d] [%d] \n' %
            (args.gpu_enc, args.gpu_dec))
    else:
        import numpy as xp
        args.gpu_enc = -1
        args.gpu_dec = -1
        sys.stderr.write('w/o using GPU\n')
    # 乱数の初期値を設定
    sys.stderr.write('# random seed [%d] \n' % (args.seed))
    np.random.seed(args.seed)
    xp.random.seed(args.seed)
    random.seed(args.seed)
    # 学習か評価か分岐
    if args.train_mode == 'train':
        chainer.global_config.train = True
        chainer.global_config.enable_backprop = True
        chainer.global_config.use_cudnn = "always"
        chainer.global_config.type_check = True
        sys.stderr.write(
            'CHAINER CONFIG  [{}] \n'.format(
                chainer.global_config.__dict__))
        if args.dropout_rate >= 1.0 or args.dropout_rate < 0.0:
            assert 0, "ERROR"
        train_model(args)
    elif args.train_mode == 'test':
        chainer.global_config.train = False
        chainer.global_config.enable_backprop = False
        chainer.global_config.use_cudnn = "always"
        chainer.global_config.type_check = True
        args.dropout_rate = 0.0
        sys.stderr.write(
            'CHAINER CONFIG  [{}] \n'.format(
                chainer.global_config.__dict__))
        ttest_model(args)
    else:
        sys.stderr.write('Please specify train or test\n')
        sys.exit(1)
