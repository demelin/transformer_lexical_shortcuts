# Project documentation

## Dependencies
- python 3.6
- TensorFlow 1.8

## Summary
Code for the reproduction of the lexical shortcut studies detailed in [Widening the Representation Bottleneck in Neural Machine Translation with Lexical Shortcuts](http://www.statmt.org/wmt19/pdf/WMT0120.pdf). Please refer to the paper for hyper-parameter settings, training and evaluation datasets used, and the primary findings. 

## Usage
Scripts used to conduct the experiments described in the paper are provided in the 'scripts' directory. Their functionality is as follows:

1. **preprocess.sh**: Used to pre-process the training, development and test corpora used in our experiments (development and test corpora first have to be converted to plain text, e.g. by using input-from-sgm.perl, provided in the Moses toolkit). Adjust as needed for different language pairs. 

2. **train.sh**: Used to train the translation models. To replicate different experiments, select the appropriate values for the *--model\_type* and *--shortcut\_type* flags (i.e. *--model\_type lexical\_shortcuts\_transformer* and *--shortcut\_type lexical\_plus\_feature\_fusion* for a transformer variant equipped with lexical shortcuts and feature-fusion). See the nmt.py file for the available options. Adding the flag *--embiggen_model* to the training script enables the transformer-BIG configuration. To use transformer-SMALL, adjust the relevant hyper-parameter values directly in the training script.

3. **test.sh**: Used to obtain the test-BLEU scores reported in the paper for each trained model. *--use_sacrebleu* returns the (more conservative) sacreBLEU score, whereas omitting this flag will return scores obtained by the script employed to calculate validation-BLEU during training (based on multi-bleu-detok.py). The latter is roughly comparable to the BLEU calculation method employed in *'Attention Is All You Need'*, Vaswani et al, 2017.

4. **train_classifier.sh**: Used to train diagnostic lexical classifiers employed in the probing studies. Enabling *--probe\_encoder* provides the classifier with access to the hidden states of the encoder, while omitting the flag trains the classifier on decoder states. *--probe_layer* denotes the ID of the encoder / decoder layer accessed by the classifier (1 being the lowest and 6 being the top-most).

5. **test_classifier.sh**: Used to obtain the accuracy of trained classifiers on a withheld test-set.

## Citation

If you find this work useful, please consider citing the accompanying paper:

```
@article{emelin2019widening,
  title={Widening the Representation Bottleneck in Neural Machine Translation with Lexical Shortcuts},
  author={Emelin, Denis and Titov, Ivan and Sennrich, Rico},
  journal={arXiv preprint arXiv:1906.12284},
  year={2019}
}
```
