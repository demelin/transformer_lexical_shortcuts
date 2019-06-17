#!/bin/sh

# This script trains the model on a pre-processed (incl. BPE) parallel corpus.

# Check if all required arguments are supplied
if [ $# -ne 1  ]; then
    echo 'Expected one argument. Exiting.'
    echo 'Usage: bash train_tr.sh run_id'
    exit 1
fi

# Source and target languages
src=en
tgt=de

# Directories
home_dir=/home/directory
main_dir=/main/directory
data_dir=$main_dir/data
train_dir=$data_dir/train
devtest_dir=$data_dir/devtest
model_dir=$main_dir/models
exp_dir=$main_dir/exp/directory

venv=$home_dir/tensorflow/environment/bin/activate
ppath=$home_dir/tensorflow/environment/bin/python3
transformer_home=$exp_dir/transformer/library/directory

script_dir=`dirname $0`
run_id=$1

# Activate python virtual environment
. $venv
# Create run-specific directory
run_dir=$exp_dir/$run_id

if [ ! -d "$run_dir" ]; then
    mkdir $run_dir
    echo "Creating $run_dir ... "
else
    echo "$run_dir already exists, either loading or overwriting its contents ... "
fi

echo "Commencing run $run_id ... "
echo "Trained model is saved to $run_dir . "

model_name=transformer

python $transformer_home/codebase/nmt.py \
    --save_to $run_dir/$model_name.npz \
    --model_name $model_name \
    --source_dataset $train_dir/corpus.bpe.$src \
    --target_dataset $train_dir/corpus.bpe.$tgt \
    --valid_source_dataset $devtest_dir/newstest2014.bpe.$src \
    --valid_target_dataset $devtest_dir/newstest2014.bpe.$tgt \
    --valid_gold_reference $devtest_dir/newstest2014.$tgt \
    --dictionaries $model_dir/joint_corpus.bpe.$src$tgt.json $model_dir/joint_corpus.bpe.$src$tgt.json \
    --model_type lexical_shortcuts_transformer \
    --shortcut_type lexical_plus_feature_fusion \
    --embedding_size 512 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --ffn_hidden_size 2048 \
    --hidden_size 512 \
    --num_heads 8 \
    --max_len -1 \
    --translation_max_len 400 \
    --sentence_batch_size 64 \
    --maxibatch_size 40 \
    --token_batch_size 3072 \
    --beam_size 4 \
    --length_normalization_alpha 0.6 \
    --disp_freq 50 \
    --valid_freq 40000 \
    --greedy_freq 40000 \
    --beam_freq 40000 \
    --save_freq 50 \
    --max_checkpoints 1000 \
    --summary_freq 50 \
    --num_gpus 4 \
    --log_file $run_dir/log.txt \
    --bleu_script $transformer_home/eval/bleu_script_sacrebleu.sh \
    --gradient_delay 2 \
    --track_grad_rates \
    --track_gate_values

