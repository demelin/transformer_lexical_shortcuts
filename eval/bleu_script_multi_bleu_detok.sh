#!/bin/bash

# Google-like BLEU computation

# Assign inputs
inputs[1]=$1  #translations
inputs[2]=$2  #references
inputs[3]=$3  #path to unprocessed gold-reference (unused)

# Assign locations
EVAL_DIR=`dirname $0`
eval_file=$EVAL_DIR/translations.txt
eval_file2=$EVAL_DIR/references.txt

for i in $(seq 1 2)
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
echo "${outputs[2]}" > $eval_file2

# Calculate BLEU
$EVAL_DIR/multi-bleu-detok.perl <(echo "${outputs[2]}") < <(echo "${outputs[1]}")




