# mlpnlp-nmt
This is a sample code of "LSTM encoder-decoder with attention mechanism" mainly for understanding a recently developed machine translation framework based on deep neural networks.

# How to use
Please see the following example.

## Data 
* Sample data from WMT15 page http://www.statmt.org/wmt16/translation-task.html

## Preprocess (vocab file)
```bash
for f in sample_data/newstest2012-4p.{en,de} ;do \
    echo ${f} ; \
    cat ${f} | sed '/^$/d' | perl -pe 's/^\s+//; s/\s+\n$/\n/; s/ +/\n/g'  | \
    LC_ALL=C sort | LC_ALL=C uniq -c | LC_ALL=C sort -r -g -k1 | \
    perl -pe 's/^\s+//; ($a1,$a2)=split;
       if( $a1 >= 3 ){ $_="$a2\t$a1\n" }else{ $_="" } ' > ${f}.vocab_t3_tab ;\
done
```
## Training 
Note that please run with ``GPU=-1`` option for no GPU environment
```bash
SLAN=de; TLAN=en; GPU=0;  EP=13 ;  \
MODEL=sample_models/filename_of_models.model ;\
python3 -u ./LSTMEncDecAttn.py -V2 \
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
```

## Evaluation
```bash
SLAN=de; GPU=0;  EP=13 ; BEAM=5 ;  \
MODEL=sample_models/filename_of_models.model ;\
python3 -u ./LSTMEncDecAttn.py \
   -t                  test \
   --gpu-enc           ${GPU} \
   --gpu-dec           ${GPU} \
   --enc-data-file     sample_data/newstest2015.h101-200.${SLAN} \
   --init-model        ${MODEL}.epoch${EP} \
   --setting           ${MODEL}.setting    \
   --beam-size         ${BEAM} \
   --max-length        150 \
   > ${MODEL}.epoch${EP}.decode_MAX${MAXLEN}_BEAM${BEAM}.txt
```
