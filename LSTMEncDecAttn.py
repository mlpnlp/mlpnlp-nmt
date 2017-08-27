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
# vocabファイルの準備例
for f in sample_data/newstest2012-4p.{en,de} ;do \
    echo ${f} ; \
    cat ${f} | sed '/^$/d' | perl -pe 's/^\s+//; s/\s+\n$/\n/; s/ +/\n/g'  | \
    LC_ALL=C sort | LC_ALL=C uniq -c | LC_ALL=C sort -r -g -k1 | \
    perl -pe 's/^\s+//; ($a1,$a2)=split;
       if( $a1 >= 3 ){ $_="$a2\t$a1\n" }else{ $_="" } ' > ${f}.vocab_t3_tab ;\
done
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
# import codecs

import chainer
import chainer.functions as chaFunc
import chainer.optimizers as chaOpt
import chainer.links as chaLink
import chainer.serializers as chaSerial
from chainer import cuda


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
        self.norm_orig = np.sqrt(chainer.optimizer._sum_sqnorm(
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
                if args.chainer_version_check[0] == 2:
                    hin = chaFunc.dropout(hout, ratio=dropout_rate)
                else:
                    hin = chaFunc.dropout(
                        hout, train=args.dropout_mode, ratio=dropout_rate)
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
        if name is not "":
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
            if args.chainer_version_check[0] == 2:
                hx = chainer.Variable(
                    self.xp.zeros((hx_shape, xs.data.shape[1], self.out_size),
                                  dtype=xs.dtype))
            else:
                hx = chainer.Variable(
                    self.xp.zeros((hx_shape, xs.data.shape[1], self.out_size),
                                  dtype=xs.dtype),
                    volatile='auto')
        return hx

    def __call__(self, hx, cx, xs, flag_train, args):
        if hx is None:
            hx = self.init_hx(xs)
        if cx is None:
            cx = self.init_hx(xs)

        # hx, cx は (layer数, minibatch数，出力次元数)のtensor
        # xsは (系列長, minibatch数，出力次元数)のtensor
        # Note: chaFunc.n_step_lstm() は最初の入力層にはdropoutしない仕様
        if args.chainer_version_check[0] == 2:
            hy, cy, ys = chaFunc.n_step_lstm(
                self.n_layers, self.dropout_rate, hx, cx, self.ws, self.bs, xs)
        else:
            hy, cy, ys = chaFunc.n_step_lstm(
                self.n_layers, self.dropout_rate, hx, cx, self.ws, self.bs, xs,
                train=flag_train, use_cudnn=self.use_cudnn)
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
            if name is not "":  # 名前を付ける
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
    def __call__(self, layer_num, input_state_list, flag_train,
                 dropout_rate, args):
        # Note: chaFunc.n_step_lstm() は最初の入力にはdropoutしない仕様なので，
        # 一層毎に手動で作った場合は手動でdropoutが必要
        if layer_num > 0:
            if args.chainer_version_check[0] == 2:
                hin = chaFunc.dropout(input_state_list, ratio=dropout_rate)
            else:
                hin = chaFunc.dropout(input_state_list,
                                      train=args.dropout_mode,
                                      ratio=dropout_rate)
        else:
            hin = input_state_list
        # layer_num層目の処理を一括で行う
        hy, cy, hout = self[layer_num](None, None, hin, flag_train, args)
        return hy, cy, hout


# EncDecの本体
class EncoderDecoderAttention:
    def __init__(self, encoderVocab, decoderVocab, setting):
        self.encoderVocab = encoderVocab  # encoderの語彙
        self.decoderVocab = decoderVocab  # decoderの語彙
        # 語彙からIDを取得するための辞書
        self.index2encoderWord = {
            v: k for k, v in six.iteritems(
                self.encoderVocab)}  # 実際はなくてもいい
        self.index2decoderWord = {
            v: k for k, v in six.iteritems(
                self.decoderVocab)}  # decoderで利用
        self.eDim = setting.eDim
        self.hDim = setting.hDim
        self.flag_dec_ifeed = setting.flag_dec_ifeed
        self.flag_enc_boseos = setting.flag_enc_boseos
        self.attn_mode = setting.attn_mode
        self.flag_merge_encfwbw = setting.flag_merge_encfwbw

        self.encVocabSize = len(encoderVocab)
        self.decVocabSize = len(decoderVocab)
        self.n_layers = setting.n_layers

    # encoder-docoderのネットワーク
    def initModel(self):
        sys.stderr.write(
            ('Vocab: enc=%d dec=%d embedDim: %d, hiddenDim: %d, '
             'n_layers: %d # [Params] dec inputfeed [%d] '
             '| use Enc BOS/EOS [%d] | attn mode [%d] '
             '| merge Enc FWBW [%d]\n'
             % (self.encVocabSize, self.decVocabSize, self.eDim, self.hDim,
                self.n_layers, self.flag_dec_ifeed,
                self.flag_enc_boseos, self.attn_mode,
                self.flag_merge_encfwbw)))
        self.model = chainer.Chain(
            # encoder embedding層
            encoderEmbed=chaLink.EmbedID(self.encVocabSize, self.eDim),
            # decoder embedding層
            decoderEmbed=chaLink.EmbedID(self.decVocabSize, self.eDim,
                                         ignore_label=-1),
            # 出力層
            decOutputL=chaLink.Linear(self.hDim, self.decVocabSize),
        )
        # logに出力する際にわかりやすくするための名前付け なくてもよい
        self.model.encoderEmbed.W.name = "encoderEmbed_W"
        self.model.decoderEmbed.W.name = "decoderEmbed_W"
        self.model.decOutputL.W.name = "decoderOutput_W"
        self.model.decOutputL.b.name = "decoderOutput_b"

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

            if not args.flag_emb_cpu:  # 指定があればCPU側のメモリ上に置く
                self.model.decoderEmbed.to_gpu(args.gpu_dec)
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
                    if args.chainer_version_check[0] == 2:
                        p.copydata(chainer.Parameter(
                            t_initializer, p.data.shape))
                    else:
                        chainer.initializers.init_weight(p.data, t_initializer)
        elif init_type == "normal":
            sys.stdout.write("# initializer is [normal] [%f]\n" % (init_scale))
            t_initializer = chainer.initializers.Normal(init_scale)
            named_params = sorted(
                optimizer.target.namedparams(),
                key=lambda x: x[0])
            for n, p in named_params:
                with cuda.get_device(p.data):
                    if args.chainer_version_check[0] == 2:
                        p.copydata(chainer.Parameter(
                            t_initializer, p.data.shape))
                    else:
                        chainer.initializers.init_weight(p.data, t_initializer)
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
            t_norm = chainer.optimizer._sum_sqnorm(p.data)
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
    # 情報を保持するためだけのクラス 主に 細切れにbackwardするための用途
    class encInfoObject:
        def __init__(self, finalHiddenVars, finalLSTMVars, encLen, cMBSize):
            self.attnList = finalHiddenVars
            self.lstmVars = finalLSTMVars
            self.encLen = encLen
            self.cMBSize = cMBSize
    ###############################################

    # encoderのembeddingを取得する関数
    def getEncoderInputEmbeddings(self, input_idx_list, args):
        # 一文一括でembeddingを取得  この方が効率が良い？
        if args.flag_emb_cpu and args.gpu_enc >= 0:
            encEmbList = chaFunc.copy(
                self.model.encoderEmbed(chainer.Variable(input_idx_list)),
                args.gpu_enc)
        else:
            xp = cuda.get_array_module(self.model.encoderEmbed.W.data)
            encEmbList = self.model.encoderEmbed(
                chainer.Variable(xp.array(input_idx_list)))
        return encEmbList

    # decoderのembeddingを取得する関数 上のgetEncoderInputEmbeddingsとほぼ同じ
    def getDecoderInputEmbeddings(self, input_idx_list, args):
        if args.flag_emb_cpu and args.gpu_dec >= 0:
            decEmbList = chaFunc.copy(
                self.model.decoderEmbed(chainer.Variable(input_idx_list)),
                args.gpu_dec)
        else:
            xp = cuda.get_array_module(self.model.decoderEmbed.W.data)
            decEmbList = self.model.decoderEmbed(
                chainer.Variable(xp.array(input_idx_list)))
        return decEmbList

    # encoder側の入力を処理する関数
    def encodeSentenceFWD(self, train_mode, sentence, args, dropout_rate):
        if args.gpu_enc != args.gpu_dec:  # encとdecが別GPUの場合
            chainer.cuda.get_device(args.gpu_enc).use()
        encLen = len(sentence)  # 文長
        cMBSize = len(sentence[0])  # minibatch size

        # 一文一括でembeddingを取得  この方が効率が良い？
        encEmbList = self.getEncoderInputEmbeddings(sentence, args)

        flag_train = (train_mode > 0)
        lstmVars = [0] * self.n_layers * 2
        if self.flag_merge_encfwbw == 0:  # fwとbwは途中で混ぜない最後で混ぜる
            hyf, cyf, fwHout = self.model.encLSTM_f(
                None, None, encEmbList, flag_train, args)  # 前向き
            hyb, cyb, bkHout = self.model.encLSTM_b(
                None, None, encEmbList[::-1], flag_train, args)  # 後向き
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
                    z, biH, flag_train, dropout_rate, args)
                # z層目後ろ向き
                hyb, cyb, bkHout = self.model.encLSTM_b(
                    z, biH[::-1], flag_train, dropout_rate, args)
                # それぞれの階層の隠れ状態およびメモリセルをデコーダに
                # 渡すために保持
                lstmVars[2 * z] = chaFunc.reshape(cyf + cyb, sp)
                lstmVars[2 * z + 1] = chaFunc.reshape(hyf + hyb, sp)
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
        # encoderの情報をencInfoObjectに集約して返す
        retO = self.encInfoObject(biHiddenStackSW01, lstmVars, encLen, cMBSize)
        return retO

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
        total_loss = chainer.Variable(xp.zeros((), dtype=xp.float32))  # 初期化
        total_loss_val = 0  # float
        correct = 0
        incorrect = 0
        proc = 0
        decoder_proc = len(decSent) - 1  # ここで処理するdecoder側の単語数

        #######################################################################
        # 1, decoder側の入力単語embeddingsをまとめて取得
        decEmbListCopy = self.getDecoderInputEmbeddings(
            decSent[:decoder_proc], args)
        decSent = xp.array(decSent)  # GPU上に移動
        #######################################################################
        # 2, decoder側のRNN部分を計算
        h4_list_copy = [0] * decoder_proc
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
            # lstm_statesをキャッシュ
            lstm_states_list_copy[index] = lstm_states
            # attentionありの場合 contextベクトルを計算
            finalHS = self.calcAttention(hOut, encInfo.attnList, aList,
                                         encInfo.encLen, cMBSize, args)
            # finalHSをキャッシュ
            h4_list_copy[index] = finalHS
        #######################################################################
        # 3, output(softmax)層の計算
        for index in reversed(six.moves.range(decoder_proc)):
            # 2で用意した copyを使って最終出力層の計算をする
            oVector = self.generateWord(h4_list_copy[index], encInfo.encLen,
                                        cMBSize, args, dropout_rate)
            # 正解データ
            correctLabel = decSent[index + 1]  # xp
            proc += (xp.count_nonzero(correctLabel + 1))
            # 必ずminibatchsizeでわる
            closs = chaFunc.softmax_cross_entropy(
                oVector, correctLabel, normalize=False)
            # これで正規化なしのloss  cf. seq2seq-attn code
            total_loss_val += closs.data * cMBSize
            if train_mode > 0:  # 学習データのみ backward する
                total_loss += closs
            # 実際の正解数を獲得したい
            t_correct = 0
            t_incorrect = 0
            # Devのときは必ず評価，学習データのときはオプションに従って評価
            if train_mode == 0 or args.doEvalAcc > 0:
                # 予測した単語のID配列 CuPy
                pred_arr = oVector.data.argmax(axis=1)
                # 正解と予測が同じなら0になるはず
                # => 正解したところは0なので，全体から引く
                t_correct = (correctLabel.size -
                             xp.count_nonzero(correctLabel - pred_arr))
                # 予測不要の数から正解した数を引く # +1はbroadcast
                t_incorrect = xp.count_nonzero(correctLabel + 1) - t_correct
            correct += t_correct
            incorrect += t_incorrect
        ####
        if train_mode > 0:  # 学習時のみ backward する
            total_loss.backward()

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
    def generateWord(self, h4, encLen, cMBSize, args, dropout_rate):
        if args.chainer_version_check[0] == 2:
            oVector = self.model.decOutputL(
                chaFunc.dropout(h4, ratio=dropout_rate))
        else:
            oVector = self.model.decOutputL(chaFunc.dropout(
                h4, train=args.dropout_mode, ratio=dropout_rate))
        return oVector


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

    def makeSentenceLenDict(self, fileName, word2indexDict, input_side=False):
        if input_side:
            d = collections.defaultdict(list)
        else:
            d = {}
        sentenceNum = 0
        sampleNum = 0
        maxLen = 0
        # ここで全てのデータを読み込む
        # TODO: codecsでないとエラーが出る環境がある？ 要調査 不要ならioにしたい
        with io.open(fileName, encoding='utf-8') as f:
            # with codecs.open(fileName, encoding='utf-8') as f:
            for sntNum, snt in enumerate(f):  # ここで全てのデータを読み込む
                snt = snt.strip()
                indexList = self.sentence2index(
                    snt, word2indexDict, input_side=input_side)
                sampleNum += len(indexList)
                if input_side:
                    # input側 ここで長さ毎でまとめたリストを作成する
                    # 値は文番号と文そのもののペア
                    d[len(indexList)].append((sntNum, indexList))
                else:
                    d[sntNum] = indexList  # decoder側 文の番号をキーとしたハッシュ
                sentenceNum += 1
                maxLen = max(maxLen, len(indexList))
        sys.stdout.write('# data sent: %10d  sample: %10d maxlen: %10d\n' % (
            sentenceNum, sampleNum, maxLen))
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
            encSentBatch.append(
                np.array([encSent for sntNum, encSent in batch],
                         dtype=np.int32).T)
            maxDecoderLength = max([len(decSentLenDict[sntNum])
                                    for sntNum, encSent in batch])
            decSentBatch.append(
                np.array([decSentLenDict[sntNum] + [-1] *
                          (maxDecoderLength - len(decSentLenDict[sntNum]))
                          for sntNum, encSent in batch], dtype=np.int32).T)
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

    def update(self, loss_stat, mbs, bc, cor, incor, tsize, proc,
               encLen, decLen):
        self.instanceNum += mbs  # 文数を数える
        self.batchCount += bc  # minibatchで何回処理したか
        self.corTot += cor
        self.incorTot += incor
        self.trainsizeTot += tsize
        self.procTot += proc
        # 強制的にGPUからCPUに値を移すため floatを利用
        self.lossVal += float(loss_stat)

        self.encMaxLen = max(encLen * mbs, self.encMaxLen)
        self.decMaxLen = max(decLen * mbs, self.decMaxLen)

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
        if args.chainer_version_check[0] == 2:
            if train_mode > 0:  # train
                chainer.global_config.train = True
                chainer.global_config.enable_backprop = True
                sys.stderr.write(
                    ('# TRAIN epoch {} drop rate={} | CHAINER CONFIG  [{}] \n'
                     .format(epoch, dropout_rate,
                             chainer.global_config.__dict__)))
            else:              # dev
                chainer.global_config.train = False
                chainer.global_config.enable_backprop = False
                sys.stderr.write(
                    ('# DEV.  epoch {} drop rate={} | CHAINER CONFIG  [{}] \n'
                     .format(epoch, dropout_rate,
                             chainer.global_config.__dict__)))
        else:
            if train_mode > 0:  # train
                args.dropout_mode = args.dropout_mode_orig
            else:              # dev
                args.dropout_mode = False
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
                decLen = len(decSent)
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
                            xp.sqrt(chainer.optimizer._sum_sqnorm(
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
                        decLen = len(decSent)
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
        encoderVocab = prepD.readVocab(args.encVocabFile)
        decoderVocab = prepD.readVocab(args.decVocabFile)
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

    # optimizerを構築
    if args.optimizer == 'SGD':
        optimizer = chaOpt.SGD(lr=args.lrate)
        sys.stdout.write(
            '# SET Learning %s: initial learning rate: %e\n' %
            (args.optimizer, optimizer.lr))
    elif args.optimizer == 'Adam':
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

    if args.optimizer == 'SGD':
        clipV = args.grad_clip
        sys.stdout.write('# USE gradient clipping: %f\n' % (clipV))
        clip_obj = Chainer_GradientClipping_rmk_v1(clipV)
        optimizer.add_hook(clip_obj)
    else:
        clipV = args.grad_clip
        sys.stdout.write('# USE gradient clipping: %f\n' % (clipV))
        clip_obj = Chainer_GradientClipping_rmk_v1(clipV)
        optimizer.add_hook(clip_obj)

    ########################################
    # 学習済みの初期モデルがあればをここで読み込む
    if args.setting_file and args.init_model_file:
        sys.stderr.write('Load model from: [%s]\n' % (args.init_model_file))
        chaSerial.load_npz(args.init_model_file, EncDecAtt.model)
    else:  # 学習済みの初期モデルがなければパラメタを全初期化する
        EncDecAtt.setInitAllParameters(optimizer, init_type=args.init_type,
                                       init_scale=args.init_scale)

    ########################################
    # ここでencoder側/decoder側のデータを全て読み込む
    if True:
        encSentLenDict = prepD.makeSentenceLenDict(
            args.encDataFile, EncDecAtt.encoderVocab, input_side=True)
        decSentLenDict = prepD.makeSentenceLenDict(
            args.decDataFile, EncDecAtt.decoderVocab, input_side=False)
        if args.mode_data_shuffle == 0:  # default
            trainData = prepD.makeBatch4Train(
                encSentLenDict,
                decSentLenDict,
                args.batch_size,
                shuffle_flag=True)
    if args.encDevelDataFile and args.decDevelDataFile:
        encSentLenDictDevel = prepD.makeSentenceLenDict(
            args.encDevelDataFile, EncDecAtt.encoderVocab, input_side=True)
        decSentLenDictDevel = prepD.makeSentenceLenDict(
            args.decDevelDataFile, EncDecAtt.decoderVocab, input_side=False)
        develData = prepD.makeBatch4Train(
            encSentLenDictDevel, decSentLenDictDevel, args.batch_size,
            shuffle_flag=False)

    prevLossDevel = 1.0e+100
    prevAccDevel = 0
    prevLossTrain = 1.0e+100
    # 学習のループ
    for epoch in six.moves.range(args.epoch):
        ####################################
        # devの評価モード
        if args.encDevelDataFile and args.decDevelDataFile:
            train_mode = 0
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
        if True:  # 学習は必ず行うことが前提
            train_mode = 1
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
        if args.outputFile:
            if (epoch + 1 == args.epoch or
                    (args.outEach != 0 and (epoch + 1) % args.outEach == 0)):
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


# 以下，評価時だけ使う関数
def updateBeamThreshold__2(queue, input):
    # list内の要素はlist,タプル，かつ，0番目の要素はスコアを仮定
    if len(queue) == 0:
        queue.append(input)
    else:
        # TODO 線形探索なのは面倒なので 効率を上げるためには要修正
        for i in six.moves.range(len(queue)):
            if queue[i][0] <= input[0]:
                continue
            tmp = queue[i]
            queue[i] = input
            input = tmp
    return queue


def decodeByBeamFast(EncDecAtt, encSent, cMBSize, max_length, beam_size, args):
    train_mode = 0  # 評価なので
    encInfo = EncDecAtt.encodeSentenceFWD(train_mode, encSent, args, 0.0)
    if args.gpu_enc != args.gpu_dec:  # encとdecが別GPUの場合
        chainer.cuda.get_device(args.gpu_dec).use()
    encLen = encInfo.encLen
    aList, finalHS = EncDecAtt.prepareDecoder(encInfo)

    idx_bos = EncDecAtt.decoderVocab['<s>']
    idx_eos = EncDecAtt.decoderVocab['</s>']
    idx_unk = EncDecAtt.decoderVocab['<unk>']

    if args.wo_rep_w:
        WFilter = xp.zeros((1, EncDecAtt.decVocabSize), dtype=xp.float32)
    else:
        WFilter = None
    beam = [(0, [idx_bos], idx_bos, encInfo.lstmVars, finalHS, WFilter)]
    dummy_b = (1.0e+100, [idx_bos], idx_bos, None, None, WFilter)

    for i in six.moves.range(max_length + 1):  # for </s>
        newBeam = [dummy_b] * beam_size

        cMBSize = len(beam)
        #######################################################################
        # beamで分割されているものを一括処理するために miniBatchとみなして処理
        # 準備としてbeamの情報を結合
        # beam内の候補をminibatchとして扱うために，axis=0 を 1から
        # cMBSizeに拡張するためにbroadcast
        biH0 = chaFunc.broadcast_to(
            encInfo.attnList, (cMBSize, encLen, EncDecAtt.hDim))
        if EncDecAtt.attn_mode == 1:
            aList_a = biH0
        elif EncDecAtt.attn_mode == 2:
            t = chaFunc.broadcast_to(
                chaFunc.reshape(
                    aList, (1, encLen, EncDecAtt.hDim)),
                (cMBSize, encLen, EncDecAtt.hDim))
            aList_a = chaFunc.reshape(t, (cMBSize * encLen, EncDecAtt.hDim))
            # TODO: 効率が悪いのでencoder側に移動したい
        else:
            assert 0, "ERROR"

        zipbeam = list(six.moves.zip(*beam))
        # axis=1 (defaultなので不要) ==> hstack
        lstm_states_a = chaFunc.concat(zipbeam[3])
        # concat(a, axis=0) == vstack(a)
        finalHS_a = chaFunc.concat(zipbeam[4], axis=0)
        # decoder側の単語を取得
        # 一つ前の予測結果から単語を取得
        wordIndex = np.array(zipbeam[2], dtype=np.int32)
        inputEmbList = EncDecAtt.getDecoderInputEmbeddings(wordIndex, args)
        #######################################################################
        hOut, lstm_states_a = EncDecAtt.processDecLSTMOneStep(
            inputEmbList, lstm_states_a, finalHS_a, args, 0.0)
        # attentionありの場合 contextベクトルを計算
        next_h4_a = EncDecAtt.calcAttention(hOut, biH0, aList_a,
                                            encLen, cMBSize, args)
        oVector_a = EncDecAtt.generateWord(next_h4_a, encLen,
                                           cMBSize, args, 0.0)
        #####
        nextWordProb_a = -chaFunc.log_softmax(oVector_a.data).data
        if args.wo_rep_w:
            WFilter_a = xp.concat(zipbeam[4], axis=0)
            nextWordProb_a += WFilter_a
        # 絶対に出てほしくない出力を強制的に選択できないようにするために
        # 大きな値をセットする
        nextWordProb_a[:, idx_bos] = 1.0e+100  # BOS
        if args.wo_unk:  # UNKは出さない設定の場合
            nextWordProb_a[:, idx_unk] = 1.0e+100

        #######################################################################
        # beam_size個だけ使う，使いたくない要素は上の値変更処理で事前に省く
        if args.gpu_enc >= 0:
            nextWordProb_a = nextWordProb_a.get()  # sort のためにCPU側に移動
        sortedIndex_a = bn.argpartition(
            nextWordProb_a, beam_size)[:, :beam_size]
        # 遅くてもbottleneckを使いたくなければ下を使う？
        # sortedIndex_a = np.argsort(nextWordProb_a)[:, :beam_size]
        #######################################################################

        for z, b in enumerate(beam):
            # まず，EOSまで既に到達している場合はなにもしなくてよい
            # (beamはソートされていることが条件)
            if b[2] == idx_eos:
                newBeam = updateBeamThreshold__2(newBeam, b)
                continue
            ##
            flag_force_eval = False
            if i == max_length:  # mode==0,1,2: free,word,char
                flag_force_eval = True

            if not flag_force_eval and b[0] > newBeam[-1][0]:
                continue
            # 3
            # 次のbeamを作るために準備
            lstm_states = lstm_states_a[:, z:z + 1, ]
            next_h4 = next_h4_a[z:z + 1, ]
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
                nb = (newProb, b[1][:] + [wordIndex], wordIndex,
                      lstm_states, next_h4, tWFilter)
                newBeam = updateBeamThreshold__2(newBeam, nb)
                continue
            # 正解が与えられている際にはこちらを使う
            # if decoderSent is not None:
            #   wordIndex = decoderSent[i]
            #   newProb =  nextWordProb[wordIndex] + b[0]
            #   if args.wo_rep_w:
            #           tWFilter = b[5].copy()
            #           tWFilter[:,wordIndex] += 1.0e+100
            #   else:
            #                tWFilter = b[5]
            #   nb = (newProb, b[1][:]+[wordIndex], wordIndex,
            #         lstm_states, next_h4, tWFilter)
            #   newBeam = updateBeamThreshold__2(newBeam, nb)
            #   continue
            # 3
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
                nb = (newProb, b[1][:] + [wordIndex], wordIndex,
                      lstm_states, next_h4, tWFilter)
                newBeam = updateBeamThreshold__2(newBeam, nb)
                #####
        ################
        # 一時刻分の処理が終わったら，入れ替える
        beam = newBeam
        if all([True if b[2] == idx_eos else False for b in beam]):
            break
        # 次の入力へ
    beam = [(b[0], b[1], b[3], b[4], [EncDecAtt.index2decoderWord[z]
                                      if z != 0
                                      else "$UNK$"
                                      for z in b[1]]) for b in beam]

    return beam


def rerankingByLengthNormalizedLoss(beam, wposi):
    beam.sort(key=lambda b: b[0] / (len(b[wposi]) - 1))
    return beam


# テスト部本体
def ttest_model(args):

    EncDecAtt = pickle.load(open(args.setting_file, 'rb'))
    EncDecAtt.initModel()
    if args.setting_file and args.init_model_file:  # モデルをここで読み込む
        sys.stderr.write('Load model from: [%s]\n' % (args.init_model_file))
        chaSerial.load_npz(args.init_model_file, EncDecAtt.model)
    else:
        assert 0, "ERROR"
    prepD = PrepareData(EncDecAtt)

    EncDecAtt.setToGPUs(args)
    sys.stderr.write('Finished loading model\n')

    sys.stderr.write('max_length is [%d]\n' % args.max_length)
    sys.stderr.write('w/o generating unk token [%r]\n' % args.wo_unk)
    sys.stderr.write('w/o generating the same words in twice [%r]\n' %
                     args.wo_rep_w)
    sys.stderr.write('beam size is [%d]\n' % args.beam_size)
    sys.stderr.write('output is [%s]\n' % args.outputFile)

    ####################################
    decMaxLen = args.max_length

    begin = time.time()
    counter = 0
    # TODO: codecsでないとエラーが出る環境がある？ 要調査 不要ならioにしたい
    with io.open(args.encDataFile, encoding='utf-8') as f:
        # with codecs.open(args.encDataFile, encoding='utf-8') as f:
        for sentence in f:
            sentence = sentence.strip()  # stripを忘れずに．．．
            # ここでは，入力された順番で一文ずつ処理する方式のみをサポート
            sourceSentence = prepD.sentence2index(
                sentence, EncDecAtt.encoderVocab, input_side=True)
            sourceSentence = np.transpose(
                np.reshape(np.array(sourceSentence, dtype=np.int32),
                           (1, len(sourceSentence))))
            # 1文ずつ処理するので，test時は基本必ずminibatch=1になる
            cMBSize = len(sourceSentence[0])
            outputBeam = decodeByBeamFast(EncDecAtt, sourceSentence, cMBSize,
                                          decMaxLen, args.beam_size, args)
            wposi = 4
            outloop = 1
            # if args.outputAllBeam > 0:
            #    outloop = args.beam_size

            # 長さに基づく正規化 このオプションを使うことを推奨
            if args.length_normalized:
                outputBeam = rerankingByLengthNormalizedLoss(outputBeam, wposi)

            for i in six.moves.range(outloop):
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
            sys.stderr.write('\rSent.Num: %5d %s  | words=%d | Time: %10.4f ' %
                             (counter, outputList, len(outputList),
                              time.time() - begin))
    sys.stderr.write('\rDONE: %5d | Time: %10.4f\n' %
                     (counter, time.time() - begin))


#######################################
# ## main関数
if __name__ == "__main__":
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
    # parser.add_argument(
    #     '--use-blank-token',
    #     dest='use_blank_token',
    #     default=0,
    #     type=int,
    #     help='use blank token for padding [int] default=0')

    # decoder options for test
    parser.add_argument(
        '--max-length',
        dest='max_length',
        default=200,
        type=int,
        help=('[decoder option] '
              'the maximum number of words in output'))
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

    # for debug
    # parser.add_argument(
    #    '--outputAllBeam',
    #    dest='outputAllBeam',
    #    default=0,
    #    type=int,
    #    help='output all candidates in beam')
    # additional options for evaluation
    # parser.add_argument(
    #     '--print-attention',
    #     dest='printAttention',
    #     default=0,
    #     type=int,
    #     help='specify whether to print the prob. of attention or not')

    ##################################################################
    # ここから開始
    sys.stderr.write('CHAINER VERSION [{}] \n'.format(chainer.__version__))
    # コマンドラインオプション取得
    args = parser.parse_args()
    # chainer version2対応のためにバージョン取得
    args.chainer_version_check = [int(z)
                                  for z in chainer.__version__.split('.')[:2]]
    if args.chainer_version_check[0] < 1 or args.chainer_version_check[0] > 2:
        assert 0, "ERROR"
    sys.stderr.write('CHAINER VERSION check [{}]\n'.format(
        args.chainer_version_check))
    # sys.stderr.write(
    #     'CHAINER CONFIG  [{}] \n'.format(chainer.global_config.__dict__))
    # プライマリのGPUをセット 或いはGPUを使わない設定にする
    if args.gpu_enc >= 0 and args.gpu_dec >= 0:
        import cupy as xp
        cuda.check_cuda_available()
        cuda.get_device(args.gpu_enc).use()
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
        if args.chainer_version_check[0] == 2:
            chainer.global_config.train = True
            chainer.global_config.enable_backprop = True
            chainer.global_config.use_cudnn = "always"
            chainer.global_config.type_check = True
            sys.stderr.write(
                'CHAINER CONFIG  [{}] \n'.format(
                    chainer.global_config.__dict__))
        else:
            args.dropout_mode_orig = True
            args.dropout_mode = True
        if args.dropout_rate >= 1.0 or args.dropout_rate < 0.0:
            assert 0, "ERROR"
        train_model(args)
    elif args.train_mode == 'test':
        if args.chainer_version_check[0] == 2:
            chainer.global_config.train = False
            chainer.global_config.enable_backprop = False
            chainer.global_config.use_cudnn = "always"
            chainer.global_config.type_check = True
            args.dropout_rate = 0.0
            sys.stderr.write(
                'CHAINER CONFIG  [{}] \n'.format(
                    chainer.global_config.__dict__))
        else:
            args.dropout_mode_orig = False
            args.dropout_mode = False
            args.dropout_rate = 0.0
        ttest_model(args)
    else:
        sys.stderr.write('Please specify train or test\n')
        sys.exit(1)
