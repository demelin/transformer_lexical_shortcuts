# !/bin/sh
# Distributed under MIT license
# this sample script preprocesses a sample corpus, including tokenization,
# truecasing, and subword segmentation.

set -x

script_dir=`dirname $0`
main_dir=$script_dir/..
# Language-independent variables (toolkit locations)
. $script_dir/invar_vars
# Language-dependent variables (corpus locations, parameters)
. $script_dir/var_vars
# Activate python virtual environment
. $venv
export PYTHONPATH=$ppath

# Tokenize training corpora
for lang in $src $tgt
    do for prefix in corpus
        do
            cat $train_dir/$prefix.$lang | \
            $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
            $moses_scripts/tokenizer/tokenizer.perl -a -l $lang -threads 16 > $train_dir/$prefix.tok.$lang
        done
    done
echo "1/10 Tokenized training corpora!"


# Tokenize development & testing corpora
for lang in $src $tgt
    do for prefix in newstest2013 newstest2014 newstest2015 newstest2016 newstest2017 newstest2018
        do
            cat $devtest_dir/$prefix.$lang | \
            $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang | \
            $moses_scripts/tokenizer/tokenizer.perl -a -l $lang > $devtest_dir/$prefix.tok.$lang
        done
    done
echo "2/10 Tokenized validation & testing corpora!"


# Clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
for prefix in corpus
    do
        $moses_scripts/training/clean-corpus-n.perl $train_dir/$prefix.tok $src $tgt $train_dir/$prefix.tok.clean 1 80
    done
echo "3/10 Cleaned-up training corpora!"


# Train truecaser
for lang in $src $tgt
    do
        $moses_scripts/recaser/train-truecaser.perl -corpus $train_dir/corpus.tok.clean.$lang -model $model_dir/truecase-model.$lang
    done
echo "4/10 Trained truecasing models!"


# Apply truecaser to cleaned training corpus
for lang in $src $tgt
    do for prefix in corpus
        do
            $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$lang < $train_dir/$prefix.tok.clean.$lang > $train_dir/$prefix.tc.$lang
        done
    done
echo "5/10 Truecased training corpora!"


# Apply truecaser to development & testing corpora
for lang in $src $tgt
    do for prefix in newstest2013 newstest2014 newstest2015 newstest2016 newstest2017 newstest2018
       do
           $moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$lang < $devtest_dir/$prefix.tok.$lang > $devtest_dir/$prefix.tc.$lang
       done
    done
echo "6/10 Truecased development & testing corpora!"


# Enable for EN->RU
# echo "Transliterating cyrillic to latin ... "
# for prefix in corpus
#     do
#         cat $train_dir/$prefix.tc.$tgt | parallel --pipe -k -j4 --block 100M "translit -t \"ISO 9\" " > $train_dir/$prefix.tc.$tgt.latin
#     done
# echo "6.5/10 Transliterated the full Russian corpus!"

# Train BPE
python $bpe_scripts/learn_joint_bpe_and_vocab.py -i $train_dir/corpus.tc.$src $train_dir/corpus.tc.$tgt.latin -o $model_dir/$src$tgt.bpe.latin -s $bpe_operations --write-vocabulary $model_dir/vocab.$src $model_dir/vocab.$tgt.latin
echo "7/10 Learned BPE encodings!"


# De-transliterate Russian BPE segments training corpus (remove version information in the latin file first)
echo "Transliterating latin BPE codes back to cyrillic ... "
mapfile -t array < $model_dir/$src$tgt.bpe.latin
echo ${array[0]} > $model_dir/bpe_version.info
sed "1d" $model_dir/$src$tgt.bpe.latin > $model_dir/$src$tgt.bpe.latin.minus_version_info
translit -t "ISO 9" -r < $model_dir/$src$tgt.bpe.latin.minus_version_info > $model_dir/$src$tgt.bpe.cyrillic
translit -t "ISO 9" -r < $model_dir/vocab.$tgt.latin > $model_dir/vocab.$tgt
# Prepend original version information to the concatenated file to ensure that it's interpreted correctly
cat $model_dir/bpe_version.info $model_dir/$src$tgt.bpe.cyrillic $model_dir/$src$tgt.bpe.latin.minus_version_info > $model_dir/$src$tgt.bpe
echo "7.5/10 De-transliterated Russian BPE segments!"


# Apply BPE to training corpora
for lang in $src $tgt
    do for prefix in corpus
        do
            if [ $lang = "ru" ]; then
                if [ $prefix = "corpus" ]; then
                    python $bpe_scripts/apply_bpe.py -c $model_dir/$src$tgt.bpe < $train_dir/$prefix.tc.$lang > $train_dir/$prefix.bpe.$lang.temp
                    python $bpe_scripts/get_vocab.py < $train_dir/$prefix.bpe.$lang.temp > $model_dir/vocab.$lang.true_counts
                    python $bpe_scripts/apply_bpe.py -c $model_dir/$src$tgt.bpe --vocabulary $model_dir/vocab.$lang.true_counts --vocabulary-threshold $bpe_threshold < $train_dir/$prefix.tc.$lang > $train_dir/$prefix.bpe.$lang
                else
                    python $bpe_scripts/apply_bpe.py -c $model_dir/$src$tgt.bpe --vocabulary $model_dir/vocab.$lang.true_counts --vocabulary-threshold $bpe_threshold < $train_dir/$prefix.tc.$lang > $train_dir/$prefix.bpe.$lang
                fi
            else
                python $bpe_scripts/apply_bpe.py -c $model_dir/$src$tgt.bpe.latin --vocabulary $model_dir/vocab.$lang --vocabulary-threshold $bpe_threshold < $train_dir/$prefix.tc.$lang > $train_dir/$prefix.bpe.$lang
            fi
        done
    done
echo "8/10 Applied BPE encoding to training corpora!"


# Apply BPE to development & testing corpora
for lang in $src $tgt
    do for prefix in newstest2013 newstest2014 newstest2015 newstest2016 newstest2017 newstest2018
        do
            if [ $lang = "ru" ]; then
                python $bpe_scripts/apply_bpe.py -c $model_dir/$src$tgt.bpe --vocabulary $model_dir/vocab.$lang.true_counts --vocabulary-threshold $bpe_threshold < $devtest_dir/$prefix.tc.$lang > $devtest_dir/$prefix.bpe.$lang
            else
                python $bpe_scripts/apply_bpe.py -c $model_dir/$src$tgt.bpe.latin --vocabulary $model_dir/vocab.$lang --vocabulary-threshold $bpe_threshold < $devtest_dir/$prefix.tc.$lang > $devtest_dir/$prefix.bpe.$lang
            fi
        done
    done
echo "9/10 Applied BPE encoding to development & testing corpora!"

# Build network dictionary
cat $train_dir/corpus.bpe.$src $train_dir/corpus.bpe.$tgt > $train_dir/joint_corpus.bpe.$src$tgt
python /path/to/naacl_transformer/codebase/build_dictionary.py $train_dir/joint_corpus.bpe.$src$tgt
mv $train_dir/joint_corpus.bpe.$src$tgt.json $model_dir
echo "10/10 Built the NMT model dictionary!"
echo "Done!"
