#!/bin/bash

# Assign inputs
inputs[1]=$1  #translations
inputs[2]=$2  #references (unused)
gold_reference=$3  #path to unprocessed gold-reference

# Assign locations
EVAL_DIR=`dirname $0`
eval_file=$EVAL_DIR/translations.txt

for i in $(seq 1)
    do
        # De-segment
        outputs[$i]="$(cat ${inputs[$i]} | \
        sed -r 's/ \@(\S*?)\@ /\1/g' | \
        sed -r 's/\@\@ //g' | \
        sed 's/&lt;s&gt;//' | \
        # De-truecase/ de-tokenize
        $EVAL_DIR/detruecase.perl | \
        $EVAL_DIR/detokenizer.perl -q)"
    done

echo "${outputs[1]}" > $eval_file

# Calculate sacreBLEU
cat $eval_file | sacrebleu $gold_reference




