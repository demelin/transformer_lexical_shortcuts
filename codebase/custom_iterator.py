import os
import gzip
import glob
import random
import logging
import tempfile
import numpy as np


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


def shuffle(files, exp_path, temporary=False):
    """ Shuffles the contents of parallel text files. """

    files = [file for file in files if file is not None]
    sorted_files = [open(file_path, 'r') for file_path in files]
    exp_path, _ = os.path.split(os.path.realpath(exp_path))
    lines = []

    # Read-in lines
    for line in sorted_files[0]:
        line = [line.strip()] + [file.readline().strip() for file in sorted_files[1:]]
        lines.append(line)

    [file.close() for file in sorted_files]

    # Shuffle
    random.shuffle(lines)

    if temporary:
        # Delete existing shuffled files
        for shuffled_old in glob.glob('{:s}/*shuf*'.format(exp_path)):
            os.remove(shuffled_old)
        # Create new shuffled files
        shuffled_files = []
        for file_path in files:
            _, filename = os.path.split(os.path.realpath(file_path))
            shuffled_files. \
                append(tempfile.NamedTemporaryFile(mode='w+', prefix='{:s}.shuf'.format(filename), dir=exp_path))
    else:
        shuffled_files = [open('{:s}.shuf'.format(file_path), 'w') for file_path in files]

    # Populate destination files with shuffled sentences while maintaining their parallel assignment
    for line in lines:
        for file_index, file in enumerate(shuffled_files):
            file.write('{:s}\n'.format(line[file_index]))

    if temporary:
        [file.seek(0) for file in shuffled_files]
    else:
        [file.close() for file in shuffled_files]

    return shuffled_files


def get_document_length(fileobject):
    """ Fast way to compute the length of a document;
     See: https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python """
    # Set up source file and buffer
    file = open(fileobject, 'rb')
    lines = 0
    buffer_size = 1024 * 1024
    read_file = file.raw.read

    # Count lines
    buffer = read_file(buffer_size)
    while buffer:
        lines += buffer.count(b'\n')
        buffer = read_file(buffer_size)
    file.close()

    return lines


class TextIterator(object):
    """Simple bi-text iterator."""

    def __init__(self,
                 config,
                 source_path,
                 target_path,
                 exp_path,
                 source_dicts,
                 target_dict,
                 sentence_batch_size,
                 token_batch_size,
                 time_major_enc=False,
                 time_major_dec=False,
                 source_vocab_sizes=None,
                 target_vocab_size=None,
                 use_factor=False,
                 skip_empty=True,
                 sort_by_length=False,
                 shuffle_each_epoch=False,
                 training=False):

        self.config = config

        self.source_path = source_path
        self.target_path = target_path
        self.exp_path = exp_path

        # Handle corpora
        if shuffle_each_epoch:
            self.source, self.target = shuffle([self.source_path, self.target_path], self.exp_path, temporary=True)
        else:
            self.source = fopen(source_path, 'r')
            if target_path is not None:
                self.target = fopen(target_path, 'r')
            else:
                self.target = None

        # Get file length (only needed when using multiple GPUs)
        self.num_lines = get_document_length(self.source_path) if self.config.num_gpus > 1 else 0
        self.all_lines_read = False

        # Handle dictionaries
        self.source_dicts = source_dicts
        self.target_dict = target_dict

        # Prune vocabulary
        if source_vocab_sizes is not None:
            for source_dict, vocab_size in zip(self.source_dicts, source_vocab_sizes):
                if vocab_size is not None and vocab_size > 0:
                    for key, token_id in source_dict.items():
                        if token_id >= vocab_size:
                            del source_dict[key]

        if target_vocab_size is not None and target_vocab_size > 0:
            for key, token_id in self.target_dict.items():
                if token_id >= target_vocab_size:
                    del self.target_dict[key]

        # Set attributes
        self.max_len = config.max_len
        # Default to dynamic batching
        if token_batch_size > 0:
            self.batch_size = token_batch_size
            self.token_batches = True
        else:
            self.batch_size = sentence_batch_size
            self.token_batches = False

        self.maxibatch_size = \
            (self.batch_size * self.config.maxibatch_size) if config.maxibatch_size > 0 else self.batch_size

        # For multi-GPU training/ inference, keep track of items remaining in a maxi-batch as well as the remaining
        # GPUs in the current GPU cycle
        self.curr_maxibatch_size = 0
        self.gpu_tracker = 0

        # Shrink batch size to accommodate beam search
        if not training:
            if config.translate_only and not config.translate_with_beam_search:
                pass
            else:
                if self.config.num_gpus > 0:
                    self.batch_size = max(1, self.batch_size // (config.beam_size // 2))

        self.time_major_enc = time_major_enc
        self.time_major_dec = time_major_dec

        self.use_factor = use_factor
        self.skip_empty = skip_empty
        self.shuffle_each_epoch = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = list()
        self.target_buffer = list()
        self.batch_overflow = list()

        self.end_of_data = False

    def __iter__(self):
        """ Returns the iterator. """
        return self

    def __next__(self):
        """ Returns batches and masks for the current train/ test step. """
        if self.target is not None:
            return self.next_train()
        else:
            return self.next_test()

    def reset(self):
        """ Resets the iteration. """
        if self.shuffle_each_epoch:
            self.source.close()
            self.target.close()
            self.source, self.target = shuffle([self.source_path, self.target_path], self.exp_path, temporary=True)
        else:
            self.source.seek(0)
            if self.target is not None:
                self.target.seek(0)
        raise StopIteration

    def get_train_batches(self):
        """ Generates a single batch to be fed into the trained model. """
        if self.end_of_data:
            self.end_of_data = False
            self.reset()

        source_batch = list()
        target_batch = list()
        source_buffer_size = 0

        # Fill buffer if empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            lines_read = 0
            try:
                for source_sentence in self.source:
                    lines_read += 1
                    source_sentence = source_sentence.split()
                    target_sentence = self.target.readline().split()

                    # Skip empty lines
                    if self.skip_empty and (len(source_sentence) == 0 or len(target_sentence) == 0):
                        continue
                    # Skip lines exceeding the maximum length
                    if self.max_len > 0 and \
                            (len(source_sentence) > self.max_len or len(target_sentence) > self.max_len):
                        continue
                    # Update buffer
                    self.source_buffer.append(source_sentence)
                    self.target_buffer.append(target_sentence)
                    # Maxi-batching
                    source_buffer_size = source_buffer_size + len(source_sentence) if self.token_batches \
                        else source_buffer_size + 1

                    # Check if all lines have been read
                    if lines_read == self.num_lines:
                        self.all_lines_read = True

                    if source_buffer_size >= self.maxibatch_size:
                        break

            except IOError:
                self.end_of_data = True

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
                self.end_of_data = False
                self.reset()

            # Sort by source/ target sentence length
            if self.sort_by_length:
                line_lengths = \
                    np.array([max(len(source_sentence), len(target_sentence)) for
                              (source_sentence, target_sentence) in zip(self.source_buffer, self.target_buffer)])
                sorted_ids = line_lengths.argsort()

                sorted_source_buffer = [self.source_buffer[sentence_id] for sentence_id in sorted_ids]
                sorted_target_buffer = [self.target_buffer[sentence_id] for sentence_id in sorted_ids]

                self.source_buffer = sorted_source_buffer
                self.target_buffer = sorted_target_buffer

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

            # Update current max-batch size
            self.curr_maxibatch_size = source_buffer_size

        # Actual work here
        # Note: <EOS> id = 0, <GO> id = 1, <UNK> id = 2
        # Track spill-over items and remaining GPUs
        spillover = False
        gpus_left = 0

        while True:
            # Check if batch size needs to be adjusted to distribute the remaining maxi-batch items among all GPUs
            curr_batch_size = self.batch_size
            if self.config.num_gpus > 1 and self.all_lines_read:
                gpus_left = self.config.num_gpus - (self.gpu_tracker % self.config.num_gpus)
                if self.curr_maxibatch_size // (gpus_left * curr_batch_size) == 0:
                    # Distribute load
                    curr_batch_size = int(np.ceil(self.curr_maxibatch_size / gpus_left))

            # Read from source file and map to word index
            try:
                source_sentence = self.source_buffer.pop()
                # Decrement the maxi-batch size
                self.curr_maxibatch_size = self.curr_maxibatch_size - len(source_sentence) if self.token_batches \
                    else self.curr_maxibatch_size - 1

                source_indices = list()
                for element in source_sentence:
                    # Account for factored inputs
                    if self.use_factor:
                        indices = [self.source_dicts[dict_id][factor] if factor in self.source_dicts[dict_id] else 2
                                   for (dict_id, factor) in enumerate(element.split('|'))]
                    else:
                        indices = [self.source_dicts[0][element] if element in self.source_dicts[0] else 2]
                    source_indices.append(indices)

                # Read from target file and map to word index
                target_sentence = self.target_buffer.pop()
                target_indices = [self.target_dict[word] if word in self.target_dict else 2 for word in target_sentence]

                source_batch.append(source_indices)
                target_batch.append(target_indices)

                longest_source = max([len(item) for item in source_batch])
                longest_target = max([len(item) for item in target_batch])

                effective_source_batch_size = \
                    len(source_batch) * longest_source if self.token_batches else len(source_batch)
                effective_target_batch_size = \
                    len(target_batch) * longest_target if self.token_batches else len(target_batch)

                if (effective_source_batch_size > curr_batch_size or effective_target_batch_size > curr_batch_size) \
                        and len(source_batch) > 1:
                    # Avoid dropping spillover maxi-batch items
                    if gpus_left == 1 and len(self.source_buffer) < self.config.num_gpus and self.all_lines_read:
                        spillover = True

                        # Add the spillover items to the current batch
                        if len(self.source_buffer) != 0:
                            continue

                    if not spillover:
                        # Push last batched sentence onto the batch overflow stack to avoid data loss
                        source_batch.pop()
                        target_batch.pop()
                        self.batch_overflow.append((source_sentence, target_sentence))
                    break

            except IndexError:
                break

        # Track available GPUs
        if self.config.num_gpus > 1:
            self.gpu_tracker += 1

        # Push batch overflow items back onto the maxi-batch
        try:
            overflow_items = self.batch_overflow.pop()
            self.source_buffer += [overflow_items[0]]
            self.target_buffer += [overflow_items[1]]
            # Increment maxi-batch size
            self.curr_maxibatch_size = self.curr_maxibatch_size + len(overflow_items[0]) \
                if self.token_batches else self.curr_maxibatch_size + 1

        except IndexError:
            pass

        return source_batch, target_batch

    def get_test_batches(self):
        """ Generates a single batch to be fed into the evaluated model. """
        if self.end_of_data:
            self.end_of_data = False
            self.reset()

        source_batch = list()
        source_buffer_size = 0

        if len(self.source_buffer) == 0:
            lines_read = 0
            try:
                for source_sentence in self.source:
                    lines_read += 1
                    source_sentence = source_sentence.split()
                    # Skip empty lines
                    if self.skip_empty and len(source_sentence) == 0:
                        continue
                    # Update buffer
                    self.source_buffer.append(source_sentence)
                    # Maxi-batching
                    source_buffer_size = source_buffer_size + len(source_sentence) if self.token_batches \
                        else source_buffer_size + 1

                    # Check if all lines have been read
                    if lines_read == self.num_lines:
                        self.all_lines_read = True

                    if source_buffer_size >= self.maxibatch_size:
                        break

            except IOError:
                self.end_of_data = True

            if len(self.source_buffer) == 0:
                self.end_of_data = False
                self.reset()

            # Sort by source/ target sentence length
            if self.sort_by_length:
                line_lengths = np.array([len(source_sentence) for source_sentence in self.source_buffer])
                sorted_ids = line_lengths.argsort()
                sorted_source_buffer = [self.source_buffer[sentence_id] for sentence_id in sorted_ids]
                self.source_buffer = sorted_source_buffer
            else:
                self.source_buffer.reverse()

            # Update current max-batch size
            self.curr_maxibatch_size = source_buffer_size

        # Actual work here
        # Note: <EOS> id = 0, <GO> id = 1, <UNK> id = 2
        # Track spill-over items and remaining GPUs
        spillover = False
        gpus_left = 0

        while True:
            # Check if batch size needs to be adjusted to distribute the remaining maxi-batch items among all GPUs
            curr_batch_size = self.batch_size
            if self.config.num_gpus > 1 and self.all_lines_read:
                gpus_left = self.config.num_gpus - (self.gpu_tracker % self.config.num_gpus)
                if self.curr_maxibatch_size // (gpus_left * curr_batch_size) == 0:
                    # Distribute load
                    curr_batch_size = int(np.ceil(self.curr_maxibatch_size / gpus_left))

            # Read from source file and map to word index
            try:
                source_sentence = self.source_buffer.pop()
                # Decrement the maxi-batch size
                self.curr_maxibatch_size = self.curr_maxibatch_size - len(source_sentence) if self.token_batches \
                    else self.curr_maxibatch_size - 1

                source_indices = list()
                for element in source_sentence:
                    # Account for factored inputs
                    if self.use_factor:
                        indices = [self.source_dicts[dict_id][factor] if factor in self.source_dicts[dict_id] else 2
                                   for (dict_id, factor) in enumerate(element.split('|'))]
                    else:
                        indices = [self.source_dicts[0][element] if element in self.source_dicts[0] else 2]
                    source_indices.append(indices)

                # Read from target file and map to word index
                source_batch.append(source_indices)
                longest_source = max([len(item) for item in source_batch])

                effective_source_batch_size = \
                    len(source_batch) * longest_source if self.token_batches else len(source_batch)

                if effective_source_batch_size > curr_batch_size and len(source_batch) > 1:
                    # Avoid dropping spillover maxi-batch items
                    if gpus_left == 1 and len(self.source_buffer) < self.config.num_gpus and self.all_lines_read:
                        spillover = True

                        # Add the spillover items to the current batch
                        if len(self.source_buffer) != 0:
                            continue

                    if not spillover:
                        # Push last batched sentence onto the batch overflow stack to avoid data loss
                        source_batch.pop()
                        self.batch_overflow.append(source_sentence)
                    break

            except IndexError:
                break

        # Track available GPUs
        if self.config.num_gpus > 1:
            self.gpu_tracker += 1

        # Push batch overflow items back onto the maxi-batch
        try:
            overflow_item = self.batch_overflow.pop()
            self.source_buffer += [overflow_item]
            # Increment maxi-batch size
            self.curr_maxibatch_size = self.curr_maxibatch_size + len(overflow_item) \
                if self.token_batches else self.curr_maxibatch_size + 1

        except IndexError:
            pass

        return source_batch

    def next_train(self):
        """ Transforms the assembled batches into padded tensor arrays at training time. """
        # Draw mini batches
        source_batch, target_batch = self.get_train_batches()
        # Calculate sentence lengths for sentences within the processed mini-batch
        source_lengths = [len(source_sentence) for source_sentence in source_batch]
        target_lengths = [len(target_sentence) for target_sentence in target_batch]
        longest_source = np.max(source_lengths) + 1  # +1s account for sentence-final <EOS> token
        longest_target = np.max(target_lengths) + 1

        # Catch empty batches
        if longest_source == 1 or longest_target == 1:
            logging.info('Empty batch!')
            # Draw mini batches
            source_batch, target_batch = self.get_train_batches()
            # Calculate sentence lengths for sentences within the processed mini-batch
            source_lengths = [len(source_sentence) for source_sentence in source_batch]
            target_lengths = [len(target_sentence) for target_sentence in target_batch]
            longest_source = np.max(source_lengths) + 1  # +1s account for sentence-final <EOS> token
            longest_target = np.max(target_lengths) + 1

        # Batch statistics
        n_samples = len(source_batch)
        n_factors = len(source_batch[0][0])
        assert n_factors == 1, 'Factored inputs are not currently supported.'

        # Batches can be time-major, i.e. have shape = [max_length, batch_size], or batch-major
        # Time-major batches are compatible with the RNN model, batch-initial ones with the transformer
        # Note sentence-final <EOS> token is appended here via initializing x and y as zero-matrices
        if self.time_major_enc:
            source = np.zeros([n_factors, longest_source, n_samples]).astype('int32')
            source_mask = np.zeros([longest_source, n_samples]).astype('float32')
        else:
            source = np.zeros([n_factors, n_samples, longest_source]).astype('int32')
            source_mask = np.zeros([n_samples, longest_source]).astype('float32')
        if self.time_major_dec:
            target = np.zeros([longest_target, n_samples]).astype('int32')
            target_mask = np.zeros([longest_target, n_samples]).astype('float32')
        else:
            target = np.zeros([n_samples, longest_target]).astype('int32')
            target_mask = np.zeros([n_samples, longest_target]).astype('float32')

        target_in = target  # <GO>, no <EOS>
        target_out = np.zeros_like(target_in)  # <EOS>, no <GO>

        for sentence_id, [source_sentence, target_sentence] in enumerate(zip(source_batch, target_batch)):
            if self.time_major_enc:
                source[:, :source_lengths[sentence_id], sentence_id] = list(zip(*source_sentence))
                source_mask[:source_lengths[sentence_id] + 1, sentence_id] = 1.
            else:
                source[:, sentence_id, :source_lengths[sentence_id]] = list(zip(*source_sentence))
                source_mask[sentence_id, :source_lengths[sentence_id] + 1] = 1.
            if self.time_major_dec:
                target_in[:target_lengths[sentence_id] + 1, sentence_id] = [1] + target_sentence  # initial <GO>
                target_out[:target_lengths[sentence_id], sentence_id] = target_sentence
                target_mask[:target_lengths[sentence_id] + 1, sentence_id] = 1.
            else:
                target_in[sentence_id, :target_lengths[sentence_id] + 1] = [1] + target_sentence  # initial <GO>
                target_out[sentence_id, :target_lengths[sentence_id]] = target_sentence
                target_mask[sentence_id, :target_lengths[sentence_id] + 1] = 1.

        source = np.squeeze(source, axis=0)
        return source, target_in, target_out, source_mask, target_mask

    def next_test(self):
        """ Transforms the assembled batches into padded tensor arrays at test time. """
        # Draw mini batches
        source_batch = self.get_test_batches()
        # Calculate sentence lengths for sentences within the processed mini-batch
        source_lengths = [len(source_sentence) for source_sentence in source_batch]
        longest_source = np.max(source_lengths) + 1  # +1s account for sentence-final <EOS> token

        # Catch empty batches
        if longest_source == 1:
            logging.info('Empty batch!')
            # Draw mini batches
            source_batch = self.get_test_batches()
            # Calculate sentence lengths for sentences within the processed mini-batch
            source_lengths = [len(source_sentence) for source_sentence in source_batch]
            longest_source = np.max(source_lengths) + 1  # +1s account for sentence-final <EOS> token

        # Batch statistics
        n_samples = len(source_batch)
        n_factors = len(source_batch[0][0])
        assert n_factors == 1, 'Factored inputs are not currently supported.'

        # Batches can be time-major, i.e. have shape = [max_length, batch_size], or batch-major
        # Time-major batches are compatible with the RNN model, batch-initial ones with the transformer
        # Note sentence-final <EOS> token is appended here via initializing x and y as zero-matrices
        if self.time_major_enc:
            source = np.zeros([n_factors, longest_source, n_samples]).astype('int32')
            source_mask = np.zeros([longest_source, n_samples]).astype('float32')
        else:
            source = np.zeros([n_factors, n_samples, longest_source]).astype('int32')
            source_mask = np.zeros([n_samples, longest_source]).astype('float32')

        for sentence_id, source_sentence in enumerate(source_batch):
            if self.time_major_enc:
                source[:, :source_lengths[sentence_id], sentence_id] = list(zip(*source_sentence))
                source_mask[:source_lengths[sentence_id] + 1, sentence_id] = 1.
            else:
                source[:, sentence_id, :source_lengths[sentence_id]] = list(zip(*source_sentence))
                source_mask[sentence_id, :source_lengths[sentence_id] + 1] = 1.

        source = np.squeeze(source, axis=0)

        # Add dummy placeholders
        target_in = np.zeros_like(source)
        target_out = np.zeros_like(source)
        target_mask = np.zeros_like(source_mask)
        return source, target_in, target_out, source_mask, target_mask
