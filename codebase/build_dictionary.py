from collections import OrderedDict
import numpy as np
import json
import sys


def main():
    """ Construct a dictionary, encoding (sub-)words as bytes. """
    for filename in sys.argv[1:]:
        print('Processing {:s}'.format(filename))
        word_counts = OrderedDict()
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_counts:
                        word_counts[w] = 0
                    word_counts[w] += 1
        words = list(word_counts.keys())
        counts = list(word_counts.values())
        sorted_idx = np.argsort(counts)
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]
        word_dict = OrderedDict()
        word_dict['<EOS>'] = 0
        word_dict['<GO>'] = 1  # added <GO>
        word_dict['<UNK>'] = 2
        for ii, ww in enumerate(sorted_words):
            word_dict[ww] = ii + 3

        with open('{:s}.json'.format(filename), 'w') as f:
            json.dump(word_dict, f, indent=2, ensure_ascii=False)

        print('Done')


if __name__ == '__main__':
    main()
