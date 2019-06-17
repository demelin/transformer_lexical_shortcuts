# Codebase accompanying the paper 'Widening the Representation Bottleneck in Neural Machine Translation with Lexical Shortcuts', (Emelin, Denis, Ivan Titov, and Rico Sennrich, Third Conference on Machine Translation, Florence, 2019.)

Scripts used to conduct the experiments described in the submitted paper are provided in the 'scripts' directory. Their functionality is as follows:

1. **preprocess.sh** is used to pre-process the training, development and test corpora used in our experiments (development and test corpora first have to be converted to plain text, e.g. by using input-from-sgm.perl, provided in the Moses toolkit). We are unable to provide the training / development / test data here due to its considerable size. However, it can be easily reproduced by running the preprocessing script on the publicly available WMT datasets. Adjust as needed for different language pairs. 

2. **train.sh** is used to train our translation systems. To replicate different experiments, select the appropriate values for the *--model\_type* and *--shortcut\_type* flags (i.e. *--model\_type lexical\_shortcuts\_transformer* and *--shortcut\_type lexical\_plus\_feature\_fusion* for a transformer variant equipped with lexical shortcuts and feature-fusion). See the nmt.py file for the available options. Adding the flag *--embiggen_model* to the training script enables the transformer-BIG configuration. To use transformer-SMALL, adjust the relevant hyper-parameter values directly in the training script.

3. **test.sh** is used to obtain the test-BLEU scores reported in the paper for each trained model. *--use_sacrebleu* returns the (usually more conservative) sacreBLEU score, whereas disabling this flag will return scores obtained by the script employed to calculate valudation-BLEU during training (based on multi-bleu-detok.py and using a processed reference). The latter is roughly comparable to the BLEU calculation method employed in 'Attention Is All You Need', Vaswani et al, 2017. Same steps as in 2 should be followed to select the desired model connfiguration.

4. **train_classifier.sh** is used to train individual lexical classifiers as done in our probing studies. Enabling *--probe\_encoder* provides the classifier with access to the hidden states of the encoder, disabling the flag trains the classifier on decoder states. *--probe_layer* denotes the ID of the encoder / decoder layer accessed by the classifier (1 being the lowest and 6 being the top-most).

5. **test_classifier.sh** is used to obtain the accuracy of trained classifiers on a witheld test-set. Same steps as in 4 should be followed to select the evaluated transformer states. We provide test files annotated with POS tags and frequency bins as part of our submission.

