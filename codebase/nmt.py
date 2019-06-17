""" Build a neural machine translation model based on the transformer architecture. """

import os
import sys
import json
import time
import logging
import argparse
import tempfile
import subprocess
import numpy as np

import tensorflow as tf

from datetime import datetime
from collections import OrderedDict

from transformer import Transformer as BaseTransformer

from lexical_shortcuts.lexical_shortcuts_transformer import Transformer as LexicalShortcutsTransformer
from lexical_shortcuts.dec_to_enc_shortcuts_transformer import Transformer as DecToEncShortcutsTransformer
from lexical_shortcuts.full_shortcuts_transformer import Transformer as FullShortcutsTransformer

from lexical_shortcuts.ablations.enc_only_shortcuts_transformer import Transformer as EncOnlyShortcutsTransformer
from lexical_shortcuts.ablations.dec_only_shortcuts_transformer import Transformer as DecOnlyShortcutsTransformer

from custom_iterator import TextIterator
from transformer_ops import get_parallel_ops, get_single_ops, VariableUpdateTrainer
from util import load_dict, seq2words, reverse_dict, get_visible_gpus, assign_to_device, count_parameters
from training_progress import TrainingProgress

# Debugging
from tensorflow.python import debug as tf_debug


def create_model(config, source_vocab_size, target_vocab_size):
    """ Creates the model independent of the TensorFlow session. """
    logging.info('Building model \'{:s}\'.'.format(config.model_name))
    # Set model-specific parameters
    if config.model_type == 'base_transformer':
        model = BaseTransformer(config, source_vocab_size, target_vocab_size, config.model_name)

    elif config.model_type == 'lexical_shortcuts_transformer':
        model = LexicalShortcutsTransformer(config, source_vocab_size, target_vocab_size, config.model_name)
    elif config.model_type == 'dec_to_enc_shortcuts_transformer':
        model = DecToEncShortcutsTransformer(config, source_vocab_size, target_vocab_size, config.model_name)
    elif config.model_type == 'full_shortcuts_transformer':
        model = FullShortcutsTransformer(config, source_vocab_size, target_vocab_size, config.model_name)

    elif config.model_type == 'enc_only_shortcuts_transformer':
        model = EncOnlyShortcutsTransformer(config, source_vocab_size, target_vocab_size, config.model_name)
    elif config.model_type == 'dec_only_shortcuts_transformer':
        model = DecOnlyShortcutsTransformer(config, source_vocab_size, target_vocab_size, config.model_name)

    else:
        raise ValueError('Model type {:s} is not supported'.format(config.model_type))

    return model


def average_checkpoints(to_load, config, sess):
    """ Averages model parameter values across the specified model checkpoints from the same training run;
    derived from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/avg_checkpoints.py """

    # Iterate over the specified checkpoints and assign them to a map
    ckpt_map = dict()
    for ckpt_path in config.reload:
        ckpt_step = ckpt_path.split('-')[-1]
        ckpt_map[int(ckpt_step)] = ckpt_path
    ckpt_steps = ckpt_map.keys()
    latest_ckpt = max(ckpt_steps)

    sorted_keys = list(ckpt_steps)
    sorted_keys.sort()

    # Use neutral weights
    scores = {ckpt_key: 1. for ckpt_key in sorted_keys}

    # Select variables to be loaded; to_load == None when training
    if to_load is None:
        to_load = {var.name: var for var in tf.global_variables()}

    # Assess checkpoints from oldest to most recent and average their values; abort if checkpoint does not exist
    var_names = to_load.keys()

    var_values = {var_name: None for var_name in var_names}
    var_dtypes = {var_name: None for var_name in var_names}

    reload_filename = ckpt_map[latest_ckpt]

    logging.info('Reading-in {:d} checkpoints and averaging parameter values.'.format(len(config.reload)))
    for ckpt_id, ckpt_key in enumerate(sorted_keys):
        logging.info('Current checkpoint: {:s} ...'.format(ckpt_map[ckpt_key]))
        # Open checkpoint
        try:
            reader = tf.contrib.framework.load_checkpoint(ckpt_map[ckpt_key])
        except tf.errors.NotFoundError:
            logging.info('Checkpoint not found. Exiting.')
            sys.exit()

        for var_name in var_names:
            var_value = reader.get_tensor(var_name)
            # Update accumulation maps
            if var_name.startswith('global_step'):
                var_values[var_name] = var_value
            else:
                var_values[var_name] = var_value * scores[ckpt_key] if var_values[var_name] is None else \
                    var_values[var_name] + (var_value * scores[ckpt_key])
                var_dtypes[var_name] = var_value.dtype

            if ckpt_id == len(sorted_keys) - 1:
                # Average collected values
                var_values[var_name] /= len(config.reload)

    logging.info('Assigning averaged values to variables.')
    assign_ops = [tf.assign(to_load[var_name], var_values[var_name]) for var_name in var_names]
    sess.run(tf.group(assign_ops))

    return reload_filename


def session_setup(config, sess, model, training=False, max_checkpoints=10):
    """ Prepares the model and auxiliary resources for operation. """
    to_init = list()
    # Exclude optimization variables to be loaded during inference (for greater model portability)
    to_load = None
    if not training:
        to_load = dict()
        model_vars = tf.global_variables()
        for var in model_vars:
            if 'optimization' in var.name:
                to_init.append(var)
            else:
                to_load[var.name.split(':')[0]] = var

    # If a stand-alone model is called, variable names don't need to be mapped
    saver = tf.train.Saver(to_load, max_to_keep=max_checkpoints)
    reload_filename = None
    no_averaging = True

    if type(config.reload) == list and len(config.reload) > 1:
        reload_filename = average_checkpoints(to_load, config, sess)
        no_averaging = False

    else:
        if config.reload is not None:
            if config.reload[0] == 'latest_checkpoint':
                checkpoint_dir = os.path.dirname(config.save_to)
                reload_filename = tf.train.latest_checkpoint(checkpoint_dir)
                if reload_filename is not None:
                    if os.path.basename(reload_filename).rsplit('-', 1)[0] != os.path.basename(config.save_to):
                        logging.error('Mismatching model filename found in the same directory while reloading '
                                      'from the latest checkpoint.')
                        sys.exit(1)
                    logging.info('Latest checkpoint found in directory {:s}.'.format(os.path.abspath(checkpoint_dir)))

            elif config.reload[0] == 'best_perplexity':
                checkpoint_dir = os.path.dirname(config.save_to)
                checkpoint_paths = tf.train.get_checkpoint_state(checkpoint_dir).all_model_checkpoint_paths
                reload_filename = [path for path in checkpoint_paths if 'best_perplexity' in path][0]
                if reload_filename is not None:
                    logging.info('Best perplexity checkpoint found in directory {:s}.'
                                 .format(os.path.abspath(checkpoint_dir)))

            elif config.reload[0] == 'best_bleu':
                checkpoint_dir = os.path.dirname(config.save_to)
                checkpoint_paths = tf.train.get_checkpoint_state(checkpoint_dir).all_model_checkpoint_paths
                reload_filename = [path for path in checkpoint_paths if 'best_bleu' in path][0]
                if reload_filename is not None:
                    logging.info('Best BLEU checkpoint found in directory {:s}.'
                                 .format(os.path.abspath(checkpoint_dir)))

            else:
                reload_filename = config.reload[0]

    # Initialize a progress tracking object and restore its values, if possible
    progress = TrainingProgress()
    progress.bad_counter = 0
    progress.uidx = 0
    progress.eidx = 0
    progress.estop = False
    progress.validation_perplexity = OrderedDict()
    progress.validation_bleu = OrderedDict()

    if reload_filename is not None and training:
        progress_path = '{:s}.progress.json'.format(reload_filename)
        if os.path.exists(progress_path):
            logging.info('Reloading training progress.')
            progress.load_from_json(progress_path)
            logging.info('Done!')
            if training:
                # If training process to be continued has been successfully completed before, terminate
                if progress.estop is True or \
                        progress.eidx > config.max_epochs or \
                        progress.uidx >= config.max_updates:
                    logging.warning('Training is already complete. Disable reloading of training progress '
                                    '(--no_reload_training_progress) or remove or modify progress file {:s} '
                                    'to train anyway.'.format(progress_path))
                    sys.exit(0)

    # If no source from which model parameters should be re-loaded has been specified, initialize model randomly
    if reload_filename is None:
        logging.info('Initializing model parameters from scratch.')
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        logging.info('Done!')
    # Otherwise, load parameters from specified source file
    else:
        reload_path = os.path.abspath(reload_filename)
        # For single checkpoint evaluation, load parameter values from checkpoint file
        if no_averaging:
            logging.info('Loading model parameters from file {:s}.'.format(reload_path))
            saver.restore(sess, reload_path)
        # Initialize optimization parameters from scratch
        if len(to_init) > 0:
            logging.info('Initializing the rest from scratch.')
            init_op = tf.variables_initializer(to_init)
            sess.run(init_op)
        # Reset global_path variable before resuming the training
        if training:
            model.load_global_step(progress.uidx, sess)
        logging.info('Done!')

    logging.info('Finished setting up the model!')

    if training:
        return saver, reload_filename, progress
    else:
        return saver, reload_filename


def load_dictionaries(config):
    """ Loads the specified dictionary files and processes them for string look-up during translation. """
    # Load in dictionaries (mapping: string -> string ID)
    source_to_index = load_dict(config.source_vocab)
    target_to_index = load_dict(config.target_vocab)

    # Truncate dictionaries, if specified
    if config.max_vocab_source > 0:
        for key, idx in source_to_index.items():
            if idx >= config.max_vocab_source:
                del source_to_index[key]

    if config.max_vocab_target > 0:
        for key, idx in target_to_index.items():
            if idx >= config.max_vocab_target:
                del target_to_index[key]

    # Reverse dictionaries (mapping: string ID -> string)
    index_to_source = reverse_dict(source_to_index)
    index_to_target = reverse_dict(target_to_index)

    # Get vocabulary sizes
    source_vocab_size = len(source_to_index.keys())
    target_vocab_size = len(target_to_index.keys())

    return source_to_index, target_to_index, index_to_source, index_to_target, source_vocab_size, target_vocab_size


def update_learning_rate(config, model_global_step):
    """ Adjust the current learning rate for the optimization of the target model based on training progress;
    As of now, specific to the transformer; see chapter 5.3. in 'Attention is all you Need'. """
    scheduled_step = \
        config.hidden_size ** (-0.5) * np.minimum((model_global_step + 1) ** (-0.5),
                                                  (model_global_step + 1) * (config.warmup_steps ** (-1.5)))
    return scheduled_step


def get_dataset_iterator(custom_iterator, num_gpus, get_handle=False):
    """ Transforms a custom iterator into a TensorFlow Dataset iterator. """
    # Create a data-set whose elements are generated by the custom iterator
    dataset = tf.data.Dataset.from_generator(lambda: custom_iterator,
                                             (tf.int32, tf.int32, tf.int32, tf.float32, tf.float32),
                                             (tf.TensorShape([None, None]),
                                              tf.TensorShape([None, None]),
                                              tf.TensorShape([None, None]),
                                              tf.TensorShape([None, None]),
                                              tf.TensorShape([None, None])))
    # Enable pre-fetching
    prefetch_value = num_gpus if num_gpus >= 1 else 1
    dataset.prefetch(prefetch_value)
    # Based on the data-set, construct an initializeable iterator
    dataset_iterator = dataset.make_initializable_iterator()
    # Optionally, generate an iterator handle
    if get_handle:
        iterator_handle = tf.placeholder(tf.string, shape=[], name='iterator_handle')
        return dataset_iterator, dataset, iterator_handle
    return dataset_iterator, dataset


def train(config, sess_config):
    """ Executes the training loop with the specified model and data sets. """
    # Prepare data
    source_to_index, target_to_index, index_to_source, index_to_target, source_vocab_size, target_vocab_size = \
        load_dictionaries(config)

    # Set-up iterators
    # Initialize text iterators
    custom_train_iterator = TextIterator(config,
                                         config.source_dataset,
                                         config.target_dataset,
                                         config.save_to,
                                         [source_to_index],
                                         target_to_index,
                                         config.sentence_batch_size,
                                         config.token_batch_size,
                                         sort_by_length=True,
                                         shuffle_each_epoch=True,
                                         training=True)

    custom_valid_iterator = TextIterator(config,
                                         config.valid_source_dataset,
                                         config.valid_target_dataset,
                                         config.save_to,
                                         [source_to_index],
                                         target_to_index,
                                         config.sentence_batch_size,
                                         config.token_batch_size,
                                         sort_by_length=False,
                                         shuffle_each_epoch=False)

    train_iterator, train_dataset, iterator_handle = \
        get_dataset_iterator(custom_train_iterator, config.num_gpus, get_handle=True)
    valid_iterator, valid_dataset = get_dataset_iterator(custom_valid_iterator, config.num_gpus)
    # Iterator initializers
    train_init_op = train_iterator.make_initializer(train_dataset)
    valid_init_op = valid_iterator.make_initializer(valid_dataset)

    # Enable handles for switching between iterators
    train_valid_iterator = tf.data.Iterator.from_string_handle(iterator_handle,
                                                               train_dataset.output_types,
                                                               train_dataset.output_shapes)

    # Set-up the model
    model = create_model(config, source_vocab_size, target_vocab_size)

    # Save model options
    config_as_dict = OrderedDict(sorted(vars(config).items()))
    json.dump(config_as_dict, open('{:s}.json'.format(config.save_to), 'w'), indent=2)

    # Initialize session
    sess = tf.Session(config=sess_config)
    if config.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root=None)
        sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    # Set up model trainer
    trainer = VariableUpdateTrainer(model,
                                    config.num_encoder_layers,
                                    train_valid_iterator,
                                    config.num_gpus,
                                    source_to_index['<EOS>'],
                                    config.gradient_delay,
                                    config.warmup_steps,
                                    config.num_gpus >= 2,
                                    sess,
                                    track_grad_rates=config.track_grad_rates,
                                    grad_norm_threshold=config.grad_norm_threshold)

    # Get validation and translation OPs
    if config.num_gpus >= 2:
        validation_ops = \
            get_parallel_ops(model, train_valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training', True)
        translation_ops = \
            get_parallel_ops(model, train_valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'translation')
        logging.info('[Parallel training, gradient delay == {:d}]'.format(config.gradient_delay))
    else:
        validation_ops = \
            get_single_ops(model, train_valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training', True)
        translation_ops = \
            get_single_ops(model, train_valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'translation')
        logging.info('[Single-device training, gradient delay == {:d}]'.format(config.gradient_delay))

    # Unpack validation and translation OPs
    _, batch_loss_op, sentence_losses_op, _ = validation_ops
    source_op, target_op, greedy_translations_op, sampled_translations_op, beam_translations_op, beam_scores_op = \
        translation_ops

    logging.info('-' * 20)
    model_size = count_parameters()
    logging.info('Number of model parameters (without activations): {:d}'.format(int(model_size)))
    logging.info('-' * 20)

    # Prepare model
    saver, checkpoint_path, progress = \
        session_setup(config, sess, model, training=True, max_checkpoints=config.max_checkpoints)

    if checkpoint_path is not None:
        logging.info('Resuming training from checkpoint {:s}'.format(checkpoint_path))

    # Handle summaries (see model definitions for summary definitions)
    train_summary_writer = None
    valid_summary_writer = None

    if config.summary_freq:
        if config.summary_dir is not None:
            summary_dir = config.summary_dir
        else:
            summary_dir = os.path.abspath(os.path.dirname(config.save_to))
        train_summary_dir = summary_dir + '/{:s}_train'.format(model.name)
        valid_summary_dir = summary_dir + '/{:s}_valid'.format(model.name)
        # Declare writers
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

    # Initialize iterator handles
    train_handle, valid_handle = sess.run([train_iterator.string_handle(), valid_iterator.string_handle()])

    # Initialize metrics
    model_global_step = 0
    training_losses = list()
    step_times = list()
    grad_norm_ratios = list()
    total_sentences, total_words = 0, 0
    early_stopped = False

    logging.info('[BEGIN TRAINING]')
    logging.info('Current global step: {:d}'.format(progress.uidx))
    logging.info('-' * 20)

    for epoch_id in range(progress.eidx, config.max_epochs):
        # Check if training has been early stopped
        if progress.estop:
            break

        # Track epoch-specific losses
        epoch_losses = list()

        logging.info('Current training epoch: {:d}'.format(epoch_id))
        logging.info('-' * 20)

        # (Re-)initialize the training iterator
        sess.run(train_init_op)

        while True:
            try:
                # Update learning rate
                learning_rate = update_learning_rate(config, model_global_step)

                # Check if summaries need to be written
                write_batch_summary = config.summary_freq and ((model_global_step % config.summary_freq == 0) or
                                                               (config.max_updates and
                                                                model_global_step % config.max_updates == 0))

                # Define feed_dict
                feed_dict = {iterator_handle: train_handle,
                             model.learning_rate: learning_rate,
                             model.training: True}

                # Update model
                batch_loss, words_processed, train_op, grad_norm_ratio, summaries = trainer.forward()
                to_fetch = [model.global_step, batch_loss, words_processed, train_op, grad_norm_ratio]

                # Optionally add summaries
                if trainer.do_update and write_batch_summary:
                    to_fetch += [summaries]

                pre_fetch_time = time.time()
                fetches = sess.run(to_fetch, feed_dict=feed_dict)
                step_times.append(time.time() - pre_fetch_time)  # Keep track of update durations

                # Skip rest of training script if gradients have been cached and not applied
                if not trainer.do_update:
                    continue

                model_global_step = fetches[0]
                training_losses += [fetches[1]]
                epoch_losses += [fetches[1]]
                total_words += fetches[2]
                grad_norm_ratios.append(fetches[4])
                # Update the persistent global step tracker
                progress.uidx = int(model_global_step)

                # Reset caches following the gradient application (not very elegant, but the only thing found to work)
                if trainer.do_update:
                    sess.run(trainer.zero_op)

                # Write summaries
                if write_batch_summary:
                    train_summary_writer.add_summary(fetches[-1], global_step=model_global_step)

                # Report progress
                if config.disp_freq and model_global_step % config.disp_freq == 0:
                    duration = sum(step_times)
                    current_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
                    logging.info('{:s}[TRAIN] Epoch {:d} | Step {:d} | Loss/ word {:4f} | Words/ sec {:.4f} | '
                                 'Words/ update {:4f} | Updates/ sec: {:.4f} | Learning rate {:.8f} | '
                                 'Grad norm ratio {:.4f}'
                                 .format(current_time, epoch_id, model_global_step,
                                         sum(training_losses) / len(training_losses),
                                         total_words / duration, total_words / len(training_losses),
                                         len(training_losses) / duration, learning_rate,
                                         sum(grad_norm_ratios) / len(grad_norm_ratios)))
                    logging.info('-' * 20)
                    step_times = list()
                    training_losses = list()
                    total_words = 0

                def sample_model_output(random_sample=False, beam_search=False, n_displayed=10):
                    """ Displays model output for greedy decoding and decoding via weighted sampling. """
                    # (Re-)initialize the validation iterator
                    sess.run(valid_init_op)
                    # Translate a single batch from the validation data-set
                    sample_feed_dict = {iterator_handle: valid_handle,
                                        model.training: False}
                    input_ops = [source_op, target_op]
                    if random_sample:
                        called_ops = [sampled_translations_op]
                        logging.info('[SAMPLED TRANSLATIONS]\n')
                    elif beam_search:
                        called_ops = [beam_translations_op, beam_scores_op]
                        logging.info('[BEAM SEARCH FOR BEAM OF {:d}]\n'.format(config.beam_size))
                    else:
                        called_ops = [greedy_translations_op]
                        logging.info('[GREEDY TRANSLATIONS]\n')

                    # Iterate over the entire validation set
                    # Ideally, only one batch should be drawn, but due to the nature of the Datatset iterator, this does
                    # not seem possible/ trivial
                    collected_fetches = list()
                    while True:
                        try:
                            sample_fetches = sess.run(input_ops + called_ops, feed_dict=sample_feed_dict)
                            collected_fetches.append(sample_fetches)
                        except tf.errors.OutOfRangeError:
                            break

                    # Surface first batch only
                    instances = zip(*collected_fetches[0])

                    for instance_id, instance in enumerate(instances):
                        logging.info('SOURCE: {:s}'.format(seq2words(instance[0], index_to_source)))
                        logging.info('TARGET: {:s}'.format(seq2words(instance[1], index_to_target)))
                        if not beam_search:
                            logging.info('SAMPLE: {:s}'.format(seq2words(instance[2], index_to_target)))
                            logging.info('\n')
                        else:
                            for sample_id, sample in enumerate(instance[2]):
                                logging.info('SAMPLE {:d}: {:s}\nScore {:.4f} | Length {:d} | Score {:.4f}'
                                             .format(sample_id, seq2words(sample, index_to_target),
                                                     instance[3][sample_id], len(sample), instance[3][sample_id]))
                                logging.info('\n')
                                # Only display top-3 translations within the beam
                                if sample_id >= 2:
                                    break
                        if instance_id >= n_displayed:
                            break

                # Monitor model performance by generating output with sampling
                if config.greedy_freq and model_global_step % config.greedy_freq == 0:
                    sample_model_output()
                    logging.info('-' * 20)

                # Monitor model performance by generating output with sampling
                if config.sample_freq and model_global_step % config.sample_freq == 0:
                    sample_model_output(random_sample=True)
                    logging.info('-' * 20)

                # Monitor model performance by generating output with beam search
                if config.beam_freq and model_global_step % config.beam_freq == 0:
                    sample_model_output(beam_search=True)
                    logging.info('-' * 20)

                if config.valid_freq and model_global_step % config.valid_freq == 0:
                    logging.info('[BEGIN VALIDATION]')
                    logging.info('-' * 20)
                    # (Re-)initialize the validation iterator
                    sess.run(valid_init_op)
                    validation_ops = [batch_loss_op, sentence_losses_op]
                    handles = [iterator_handle, valid_handle]

                    # Get validation perplexity only
                    validation_loss, validation_perplexity, _, validation_global_step = \
                        validation_loop(sess, model, validation_ops, handles, valid_summary_writer)

                    # Optionally calculate validation BLEU
                    if config.bleu_script is not None:
                        # Re-initialize the validation iterator
                        sess.run(valid_init_op)
                        decoding_ops = [target_op, greedy_translations_op, beam_translations_op, beam_scores_op]
                        validation_bleu = \
                            validation_bleu_loop(sess, model, config, decoding_ops, handles, index_to_target,
                                                 valid_summary_writer, validation_global_step)

                        # Save best-BLEU checkpoints
                        if len(progress.validation_bleu) == 0 or \
                                validation_bleu > max(list(progress.validation_bleu.values())):
                            progress.validation_bleu[int(model_global_step)] = validation_bleu

                            saver.save(sess, save_path='{:s}-best_bleu'.format(config.save_to))
                            logging.info(
                                '[CHECKPOINT] Saved a best-BLEU model checkpoint to {:s}.'.format(config.save_to))
                            progress_path = '{:s}-best_bleu.progress.json'.format(config.save_to)
                            progress.save_to_json(progress_path)
                            logging.info('-' * 20)
                        else:
                            # Track BLEU
                            progress.validation_bleu[int(model_global_step)] = validation_bleu

                    if len(progress.validation_perplexity) == 0 or \
                            validation_perplexity < min(list(progress.validation_perplexity.values())):
                        progress.validation_perplexity[int(model_global_step)] = validation_perplexity

                        # Save model checkpoint in case validation performance has improved
                        saver.save(sess, save_path='{:s}-best_perplexity'.format(config.save_to))
                        logging.info(
                            '[CHECKPOINT] Saved a best-perplexity model checkpoint to {:s}.'.format(config.save_to))
                        progress_path = '{:s}-best_perplexity.progress.json'.format(config.save_to)
                        progress.save_to_json(progress_path)
                        logging.info('-' * 20)
                        progress.bad_counter = 0
                    else:
                        # Track perplexity
                        progress.validation_perplexity[int(model_global_step)] = validation_perplexity

                        # Check for early-stopping
                        progress.bad_counter += 1
                        if progress.bad_counter > config.patience > 0:
                            # Execute early stopping of the training
                            logging.info(
                                'No improvement observed on the validation set for {:d} steps. Early stop!'
                                    .format(progress.bad_counter))
                            progress.estop = True
                            early_stopped = True
                            break

                # Save model parameters
                if config.save_freq and model_global_step % config.save_freq == 0:
                    saver.save(sess, save_path=config.save_to, global_step=model_global_step)
                    logging.info(
                        '[CHECKPOINT] Saved a scheduled model checkpoint to {:s}.'.format(config.save_to))
                    logging.info('-' * 20)
                    progress_path = '{:s}-{:d}.progress.json'.format(config.save_to, model_global_step)
                    progress.save_to_json(progress_path)

                if config.max_updates and model_global_step % config.max_updates == 0:
                    logging.info('Maximum number of updates reached!')
                    saver.save(sess, save_path=config.save_to, global_step=progress.uidx)
                    logging.info('[CHECKPOINT] Saved the training-final model checkpoint to {:s}.'
                                 .format(config.save_to))
                    logging.info('-' * 20)
                    progress.estop = True
                    progress_path = '{:s}-{:d}.progress.json'.format(config.save_to, progress.uidx)
                    progress.save_to_json(progress_path)
                    break

            except tf.errors.OutOfRangeError:
                trainer.curr_agg_step -= 1
                break

        if not early_stopped:
            logging.info('Epoch {:d} concluded'.format(epoch_id))
            try:
                logging.info('Average epoch loss: {:.4f}.'.format(sum(epoch_losses) / len(epoch_losses)))
            except ZeroDivisionError:
                pass
            # Update the persistent global step tracker
            progress.uidx = int(model_global_step)
            # Update the persistent epoch tracker
            progress.eidx += 1

    # Close active session
    sess.close()


def validation_loop(sess, model, ops, handles, valid_summary_writer, external=False):
    """ Iterates over the validation data, calculating a trained model's cross-entropy. """
    # Unpack OPs
    batch_loss_op, sentence_losses_op = ops

    # Initialize metrics
    valid_losses = list()
    sentence_losses = list()
    valid_global_step = 0

    # Unpack iterator variables
    if handles is not None:
        handle, valid_handle = handles
        feed_dict = {handle: valid_handle,
                     model.training: False}
    else:
        feed_dict = {model.training: False}

    logging.info('Estimating validation loss ... ')
    while True:
        try:
            # Run a forward pass through the model
            # Note, per-sentence losses used by the model are already length-normalized
            fetches = sess.run([model.global_step, batch_loss_op, sentence_losses_op], feed_dict=feed_dict)

            if fetches is not None:
                valid_losses += [fetches[1]]
                sentence_losses += fetches[2].tolist()
                valid_global_step = fetches[0]
                if len(sentence_losses) > 0:
                    logging.info('Evaluated {:d} sentences'.format(len(sentence_losses)))

        except tf.errors.OutOfRangeError:
            break

    # Report
    total_valid_loss = sum(valid_losses)
    mean_valid_loss = total_valid_loss / len(valid_losses)
    valid_perplexity = np.exp(mean_valid_loss)
    if not external:
        current_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
        logging.info('-' * 20)
        logging.info('{:s}[VALID] Loss/ word {:.4f} | Perplexity: {:.4f} | Sentence total {:d}'
                     .format(current_time, mean_valid_loss, valid_perplexity, len(sentence_losses)))

    # Write summaries
    if valid_summary_writer:
        valid_loss_summary = \
            tf.Summary(value=[tf.Summary.Value(tag='validation_loss', simple_value=mean_valid_loss)])
        valid_perplexity_summary = \
            tf.Summary(value=[tf.Summary.Value(tag='validation_perplexity', simple_value=valid_perplexity)])
        valid_summary_writer.add_summary(valid_loss_summary, global_step=valid_global_step)
        valid_summary_writer.add_summary(valid_perplexity_summary, global_step=valid_global_step)

    return mean_valid_loss, valid_perplexity, sentence_losses, valid_global_step


def validation_bleu_loop(sess, model, config, ops, handles, target_dict, valid_summary_writer, valid_global_step,
                         external=False):
    """ Iterates over the validation data, calculating the BLEU score of a trained model's beam-search translations. """
    # Unpack iterator variables
    if handles is not None:
        handle, valid_handle = handles
        feed_dict = {handle: valid_handle,
                     model.training: False}
    else:
        feed_dict = {model.training: False}

    logging.info('Estimating validation BLEU ... ')
    temp_translation_file = tempfile.NamedTemporaryFile(mode='w')
    temp_reference_file = tempfile.NamedTemporaryFile(mode='w')
    # Generate validation set translations
    translation_loop(sess,
                     ops,
                     feed_dict,
                     target_dict,
                     temp_translation_file,
                     temp_reference_file,
                     external=False,
                     beam_decoding=True,
                     full_beam=False)

    # Assumes multi_bleu_detok.perl is used for BLEU calculation and reporting
    temp_translation_file.flush()
    temp_reference_file.flush()
    process_args = \
        [config.bleu_script, temp_translation_file.name, temp_reference_file.name, config.valid_gold_reference]
    process = subprocess.Popen(process_args, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    bleu_score = 0.0
    if len(stderr) > 0:
        logging.warning('Validation script wrote the following to standard error:\n{}'.format(stderr))
    if process.returncode != 0:
        logging.warning('Validation script failed (returned exit status of {:d})'.format(process.returncode))
    try:
        print('Validation script output:\n{}'.format(stdout))
        if config.use_sacrebleu:
            bleu_score = float(stdout.decode('utf-8').split(' = ')[1].split(' ')[0])
        else:
            bleu_score = float(stdout.decode('utf-8').split(' ')[2][:-1])
    except IndexError:
        logging.warning('Unable to extract validation-BLEU from the script output.'.format(stdout))

    # Report
    if not external:
        current_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
        logging.info('-' * 20)
        logging.info('{:s}[VALID] BLEU: {:.2f}'.format(current_time, bleu_score))

    # Write summaries
    if valid_summary_writer and valid_global_step:
        valid_loss_summary = \
            tf.Summary(value=[tf.Summary.Value(tag='validation_bleu', simple_value=bleu_score)])
        valid_summary_writer.add_summary(valid_loss_summary, global_step=valid_global_step)
    return bleu_score


def translation_loop(sess, ops, feed_dict, target_dict, out_file, ref_file=None, external=False, beam_decoding=False,
                     full_beam=False):
    """ Iterates over the translation source, generating translations in the target language. """
    # Unpack OPs
    target_op, greedy_trans_op, beam_trans_op, beam_scores_op = ops

    # Track progress
    total_sentences = 0
    translations = list()
    references = list()
    beam_scores = list()
    start_time = time.time()

    while True:
        try:
            if beam_decoding:
                ref_batch, target_batch, scores = \
                    sess.run([target_op, beam_trans_op, beam_scores_op], feed_dict=feed_dict)
            else:
                ref_batch, target_batch = sess.run([target_op, greedy_trans_op], feed_dict=feed_dict)
                scores = None
            if target_batch is not None:
                translations.append(list(target_batch))
                references.append(list(ref_batch))
                if scores is not None:
                    beam_scores.append(list(scores))
                total_sentences += target_batch.shape[0]
                if len(translations) > 0:
                    logging.info('Translated {:d} sentences'.format(total_sentences))

        except tf.errors.OutOfRangeError:
            break

    duration = time.time() - start_time

    # Flatten information to be printed
    if beam_decoding:
        output_beams = list()
        score_beams = list()
        for batch_id, translation_batch in enumerate(translations):
            output_beams += [beams for beams in translation_batch]  # unpack batches
            score_beams += [beams for beams in beam_scores[batch_id]]
        outputs = list(zip(output_beams, score_beams))
    else:
        outputs = [sentence for batch in translations for sentence in batch]
        outputs = np.array(outputs, dtype=np.object)
    # Flatten references
    references = [sentence for batch in references for sentence in batch]
    references = np.array(references, dtype=np.object)

    # Write translations to file
    for sentence_id in range(len(outputs)):
        if beam_decoding:
            beams = list(zip(outputs[sentence_id][0], outputs[sentence_id][1]))
            best_sequence, score = beams[0]
            target_string = '{:s}\n'.format(seq2words(best_sequence, target_dict))
            # if external:
            #     # Write scores
            #     target_string = '{:s} | {:.4f}\n'.format(target_string.strip(), score)
            out_file.write(target_string)
            if full_beam:
                # Write the full beam
                for sequence, score in beams[1:]:
                    target_string = seq2words(sequence, target_dict)
                    out_file.write('{:s} | {:.4f}\n'.format(target_string, score))
                out_file.write('\n')
        else:
            target_string = seq2words(outputs[sentence_id], target_dict)
            out_file.write('{:s}\n'.format(target_string))
        # Write references
        if ref_file:
            ref_string = seq2words(references[sentence_id], target_dict)
            ref_file.write('{:s}\n'.format(ref_string))

    if external:
        # Report to STDOUT
        logging.info('-' * 20)
        logging.info('Translated {:d} sentences in {:.4f} seconds at {:.4f} sentences per second.'
                     .format(total_sentences, duration, total_sentences / duration))


def validate(config, sess_config):
    """ Helper function for executing model validation outside of the training loop. """

    assert config.reload is not None, \
        'Model path is not specified. Set path to model checkpoint using the --reload flag.'

    # Prepare data
    source_to_index, target_to_index, index_to_source, index_to_target, source_vocab_size, target_vocab_size = \
        load_dictionaries(config)
    # Set-up iterator
    custom_valid_iterator = TextIterator(config,
                                         config.valid_source_dataset,
                                         config.valid_target_dataset,
                                         config.save_to,
                                         [source_to_index],
                                         target_to_index,
                                         config.sentence_batch_size,
                                         config.token_batch_size,
                                         sort_by_length=False,
                                         shuffle_each_epoch=False)

    valid_iterator, _ = get_dataset_iterator(custom_valid_iterator, config.num_gpus)

    # Set-up the model
    model = create_model(config, source_vocab_size, target_vocab_size)

    # Get model OPs
    if config.num_gpus >= 2:
        validation_ops = get_parallel_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training')
        translation_ops = \
            get_parallel_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'translation')
        logging.info('[Parallel validation]')
    else:
        validation_ops = get_single_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training')
        translation_ops = \
            get_single_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'translation')
        logging.info('[Single-device validation]')

    # Unpack OPs
    _, batch_loss_op, sentence_losses_op, _, summaries_op = validation_ops
    source_op, target_op, greedy_translations_op, sampled_translations_op, beam_translations_op, beam_scores_op = \
        translation_ops

    # Initialize session
    sess = tf.Session(config=sess_config)

    # Prepare model
    saver, checkpoint_path = session_setup(config, sess, model, training=False)
    logging.info('-' * 20)
    if checkpoint_path is not None:
        logging.info('Validating model initialized form checkpoint {:s}'.format(checkpoint_path))
    else:
        logging.info('No checkpoint to initialize the translation model from could be found. Exiting.')
        sys.exit(1)

    logging.info('-' * 20)
    logging.info('Performing validation on corpus {:s}'.format(config.valid_target_dataset, model.name))
    logging.info('[BEGIN VALIDATION]')
    logging.info('-' * 20)

    # Validate
    sess.run(valid_iterator.initializer)
    valid_ops = [batch_loss_op, sentence_losses_op]
    valid_loss, valid_perplexity, sentence_losses, _ = \
        validation_loop(sess, model, valid_ops, None, None, external=True)

    logging.info('-' * 20)

    # Calculate BLEU
    sess.run(valid_iterator.initializer)
    translation_ops = [target_op, greedy_translations_op, beam_translations_op, beam_scores_op]
    valid_bleu = \
        validation_bleu_loop(sess, model, config, translation_ops, None, index_to_target, None, None, external=True)

    # Report
    corpus_lines = open(config.valid_target_dataset).readlines()
    logging.info('-' * 20)
    for line, cost in zip(corpus_lines, sentence_losses):
        logging.info('{:s} | {:.4f}'.format(line.strip(), cost))
    logging.info('-' * 20)

    mean_valid_loss = sum(sentence_losses) / len(sentence_losses)
    valid_perplexity = np.exp(mean_valid_loss)
    logging.info('Loss/ word: {:.4f} | Perplexity: {:.4f} | BLEU: {:.4f}'
                 .format(mean_valid_loss, valid_perplexity, valid_bleu))


def translate(config, sess_config, model=None):
    """ Produces translations of the specified corpus using a trained translation model. """
    if model is not None:
        assert config.reload is not None, \
            'Model path is not specified. Set path to model checkpoint using the --reload flag.'

    # Prepare data
    source_to_index, target_to_index, index_to_source, index_to_target, source_vocab_size, target_vocab_size = \
        load_dictionaries(config)

    # Set-up iterator
    custom_translate_iterator = TextIterator(config,
                                             config.translate_source_file,
                                             None,
                                             config.save_to,
                                             [source_to_index],
                                             target_to_index,
                                             config.sentence_batch_size,
                                             config.token_batch_size,
                                             sort_by_length=False,
                                             shuffle_each_epoch=False)

    translate_iterator, _ = get_dataset_iterator(custom_translate_iterator, config.num_gpus)

    # Set-up the model
    model = create_model(config, source_vocab_size, target_vocab_size)

    # For now, default to single-device OP; TODO: Fix for multi-GPU in the future.
    translation_ops = \
        get_single_ops(model, translate_iterator, config.num_gpus, source_to_index['<EOS>'], 'translation')
    logging.info('[Single-device translation]')

    # Unpack OPs
    _, target_op, greedy_translations_op, _, beam_translations_op, beam_scores_op = translation_ops

    # Initialize session
    sess = tf.Session(config=sess_config)

    # Prepare model
    saver, checkpoint_path = session_setup(config, sess, model, training=False)
    logging.info('-' * 20)
    if checkpoint_path is not None:
        logging.info('Translation model initialized form checkpoint {:s}'.format(checkpoint_path))
        if len(config.reload) > 1:
            logging.info('... averaged over {:d} preceding checkpoints.'.format(len(config.reload)))
    else:
        logging.info('No checkpoint to initialize the translation model from could be found. Exiting.')
        sys.exit(1)

    logging.info('-' * 20)
    logging.info('NOTE: Maximum translation length is capped to {:d}.'.format(config.translation_max_len))
    logging.info('Translating {:s} to {:s}.'.format(config.translate_source_file, config.translate_target_file))
    logging.info('-' * 20)

    # Define the feed_dict for the translation loop
    feed_dict = {model.training: False}

    # Open target file
    target_file = open(config.translate_target_file, 'w')

    # Initialize the inference iterator
    sess.run(translate_iterator.initializer)

    # Translate the source data-set
    translation_loop(sess,
                     [target_op, greedy_translations_op, beam_translations_op, beam_scores_op],
                     feed_dict,
                     index_to_target,
                     target_file,
                     external=True,
                     beam_decoding=config.translate_with_beam_search,
                     full_beam=config.full_beam)

    target_file.close()


def translation_scorer(config, sess_config):
    """ Helper function for scoring individual test-set translations, as required for the evaluation of ablations
    corpora such as LingEval97 and ContraWSD. """

    assert config.reload is not None, \
        'Model path is not specified. Set path to model checkpoint using the --reload flag.'

    # Prepare data
    source_to_index, target_to_index, index_to_source, index_to_target, source_vocab_size, target_vocab_size = \
        load_dictionaries(config)
    # Set-up iterator
    custom_valid_iterator = TextIterator(config,
                                         config.valid_source_dataset,
                                         config.valid_target_dataset,
                                         config.save_to,
                                         [source_to_index],
                                         target_to_index,
                                         config.sentence_batch_size,
                                         config.token_batch_size,
                                         sort_by_length=False,
                                         shuffle_each_epoch=False)

    valid_iterator, _ = get_dataset_iterator(custom_valid_iterator, config.num_gpus)

    # Set-up the model
    model = create_model(config, source_vocab_size, target_vocab_size)

    # Get model OPs
    if config.num_gpus >= 2:
        validation_ops = get_parallel_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training')
    else:
        validation_ops = get_single_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training')

    # Unpack OPs
    _, batch_loss_op, sentence_losses_op, _, summaries_op = validation_ops

    # Initialize session
    sess = tf.Session(config=sess_config)

    # Prepare model
    saver, checkpoint_path = session_setup(config, sess, model, training=False)
    logging.info('-' * 20)
    if checkpoint_path is not None:
        logging.info('Scoring validation set sentences for the model initialized form checkpoint {:s}'
                     .format(checkpoint_path))
    else:
        logging.info('No checkpoint to initialize the translation model from could be found. Exiting.')
        sys.exit(1)

    logging.info('-' * 20)
    logging.info('Scoring validation set sentences in corpus {:s}'.format(config.valid_target_dataset, model.name))
    logging.info('-' * 20)

    # Collect sentence scores
    sess.run(valid_iterator.initializer)
    feed_dict = {model.training: False}
    all_sentence_scores = list()
    sentence_id = 0

    while True:
        try:
            sentence_losses = sess.run(sentence_losses_op, feed_dict=feed_dict)
            all_sentence_scores += sentence_losses.tolist()

            if (sentence_id + 1) % 100 == 0:
                logging.info('Collected model scores for {:d} sentences'.format(sentence_id + 1))
            sentence_id += 1

        except tf.errors.OutOfRangeError:
            break

    logging.info('Done')

    # Write to file
    destination_dir = '.'.join(config.valid_source_dataset.split('.')[: -1])
    destination_path = '{:s}.{:s}.scores'.format(destination_dir, config.model_type)
    with open(destination_path, 'w') as dst:
        for score in all_sentence_scores:
            dst.write('{:f}\n'.format(score))
    logging.info('Scores file saved to {:s}'.format(destination_path))


def parse_args():
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('data sets; model loading and saving')
    data.add_argument('--source_dataset', type=str, metavar='PATH',
                      help='parallel training corpus (source)')
    data.add_argument('--target_dataset', type=str, metavar='PATH',
                      help='parallel training corpus (target)')
    data.add_argument('--dictionaries', type=str, required=True, metavar='PATH', nargs='+',
                      help='model vocabularies (source & target)')
    data.add_argument('--max_vocab_source', type=int, default=-1, metavar='INT',
                      help='maximum length of the source vocabulary; unlimited by default (default: %(default)s)')
    data.add_argument('--max_vocab_target', type=int, default=-1, metavar='INT',
                      help='maximum length of the target vocabulary; unlimited by default (default: %(default)s)')

    network = parser.add_argument_group('network parameters')
    network.add_argument('--model_name', type=str, default='nematode_model',
                         help='model file name (default: %(default)s)')
    network.add_argument('--model_type', type=str, default='transformer',
                         choices=['base_transformer',
                                  'lexical_shortcuts_transformer',
                                  'dec_to_enc_shortcuts_transformer',
                                  'full_shortcuts_transformer',
                                  'enc_only_shortcuts_transformer',
                                  'dec_only_shortcuts_transformer'],
                         help='type of the model to be trained / used for inference (default: %(default)s)')
    network.add_argument('--embiggen_model', action='store_true',
                         help='scales up the model to match the transformer-BIG specifications')
    network.add_argument('--embedding_size', type=int, default=512, metavar='INT',
                         help='embedding layer size (default: %(default)s)')
    network.add_argument('--num_encoder_layers', type=int, default=6, metavar='INT',
                         help='number of encoder layers')
    network.add_argument('--num_decoder_layers', type=int, default=6, metavar='INT',
                         help='number of decoder layers')
    network.add_argument('--ffn_hidden_size', type=int, default=2048, metavar='INT',
                         help='inner dimensionality of feed-forward sub-layers in FAN models (default: %(default)s)')
    network.add_argument('--hidden_size', type=int, default=512, metavar='INT',
                         help='dimensionality of the model\'s hidden representations (default: %(default)s)')
    network.add_argument('--num_heads', type=int, default=8, metavar='INT',
                         help='number of attention heads used in multi-head attention (default: %(default)s)')
    network.add_argument('--untie_decoder_embeddings', action='store_true',
                         help='untie the decoder embedding matrix from the output projection matrix')
    network.add_argument('--untie_enc_dec_embeddings', action='store_true',
                         help='untie the encoder embedding matrix from the embedding and '
                              'projection matrices in the decoder')

    training = parser.add_argument_group('training parameters')
    training.add_argument('--max_len', type=int, default=100, metavar='INT',
                          help='maximum sequence length for training and validation (default: %(default)s)')
    training.add_argument('--token_batch_size', type=int, default=4096, metavar='INT',
                          help='mini-batch size in tokens; set to 0 to use sentence-level batch size '
                               '(default: %(default)s)')
    training.add_argument('--sentence_batch_size', type=int, default=64, metavar='INT',
                          help='mini-batch size in sentences (default: %(default)s)')
    training.add_argument('--maxibatch_size', type=int, default=20, metavar='INT',
                          help='maxi-batch size (number of mini-batches sorted by length) (default: %(default)s)')
    training.add_argument('--max_epochs', type=int, default=100, metavar='INT',
                          help='maximum number of training epochs (default: %(default)s)')
    training.add_argument('--max_updates', type=int, default=1000000, metavar='INT',
                          help='maximum number of updates (default: %(default)s)')
    training.add_argument('--warmup_steps', type=int, default=4000, metavar='INT',
                          help='number of initial updates during which the learning rate is increased linearly during '
                               'learning rate scheduling(default: %(default)s)')
    training.add_argument('--learning_rate', type=float, default=2e-4, metavar='FLOAT',
                          help='initial learning rate (default: %(default)s)')
    training.add_argument('--adam_beta1', type=float, default=0.9, metavar='FLOAT',
                          help='exponential decay rate of the mean estimate (default: %(default)s)')
    training.add_argument('--adam_beta2', type=float, default=0.98, metavar='FLOAT',
                          help='exponential decay rate of the variance estimate (default: %(default)s)')
    training.add_argument('--adam_epsilon', type=float, default=1e-9, metavar='FLOAT',
                          help='prevents division-by-zero (default: %(default)s)')
    training.add_argument('--dropout_embeddings', type=float, default=0.1, metavar='FLOAT',
                          help='dropout applied to sums of word embeddings and positional encodings '
                               '(default: %(default)s)')
    training.add_argument('--dropout_residual', type=float, default=0.1, metavar='FLOAT',
                          help='dropout applied to residual connections (default: %(default)s)')
    training.add_argument('--dropout_relu', type=float, default=0.1, metavar='FLOAT',
                          help='dropout applied to the internal activation of the feed-forward sub-layers '
                               '(default: %(default)s)')
    training.add_argument('--dropout_attn', type=float, default=0.1, metavar='FLOAT',
                          help='dropout applied to attention weights (default: %(default)s)')
    training.add_argument('--label_smoothing_discount', type=float, default=0.1, metavar='FLOAT',
                          help='discount factor for regularization via label smoothing (default: %(default)s)')
    training.add_argument('--grad_norm_threshold', type=float, default=0., metavar='FLOAT',
                          help='gradient clipping threshold - may improve training stability; '
                               'disabled by default (default: %(default)s)')
    training.add_argument('--save_freq', type=int, default=5000, metavar='INT',
                          help='save frequency (default: %(default)s)')
    training.add_argument('--save_to', type=str, default='model', metavar='PATH',
                          help='model checkpoint location (default: %(default)s)')
    training.add_argument('--reload', type=str, nargs='+', default=None, metavar='PATH',
                          help='load existing model from this path; set to \'latest_checkpoint\' '
                               'to reload the latest checkpoint found in the --save_to directory')
    training.add_argument('--max_checkpoints', type=int, default=1000, metavar='INT',
                          help='number of checkpoints to keep (default: %(default)s)')
    training.add_argument('--summary_dir', type=str, required=False, metavar='PATH',
                          help='directory for saving summaries (default: same as --save_to)')
    training.add_argument('--summary_freq', type=int, default=100, metavar='INT',
                          help='summary writing frequency; 0 disables summaries (default: %(default)s)')
    training.add_argument('--num_gpus', type=int, default=0, metavar='INT',
                          help='number of GPUs to be used by the system; '
                               'no GPUs are used by default (default: %(default)s)')
    training.add_argument('--log_file', type=str, default=None, metavar='PATH',
                          help='log file location (default: %(default)s)')
    training.add_argument('--debug', action='store_true',
                          help='enable the TF debugger')
    training.add_argument('--shortcut_type', type=str, default='lexical',
                          choices=['lexical', 'lexical_plus_feature_fusion', 'non-lexical'],
                          help='defines the shortcut variant to use in the version of the transformer equipped with '
                               'shortcut connections')
    training.add_argument('--gradient_delay', type=int, default=0, metavar='INT',
                          help='Amount of steps by which the optimizer updates are to be delayed; '
                               'longer delays correspond to larger effective batch sizes (default: %(default)s)')
    training.add_argument('--track_grad_rates', action='store_true',
                          help='track gradient norm rates and parameter-grad rates as TensorBoard summaries')
    training.add_argument('--track_gate_values', action='store_true',
                          help='track gate activations for models with shortcuts as TensorBoard summaries')

    validation = parser.add_argument_group('validation parameters')
    validation.add_argument('--valid_source_dataset', type=str, default=None, metavar='PATH',
                            help='source validation corpus (default: %(default)s)')
    validation.add_argument('--valid_target_dataset', type=str, default=None, metavar='PATH',
                            help='target validation corpus (default: %(default)s)')
    validation.add_argument('--valid_gold_reference', type=str, default=None, metavar='PATH',
                            help='unprocessed target validation corpus used in calculating sacreBLEU '
                                 '(default: %(default)s)')
    validation.add_argument('--use_sacrebleu', action='store_true',
                            help='whether to use sacreBLEU for validation and testing')
    validation.add_argument('--valid_freq', type=int, default=4000, metavar='INT',
                            help='validation frequency (default: %(default)s)')
    validation.add_argument('--patience', type=int, default=-1, metavar='INT',
                            help='number of steps without validation-loss improvement required for early stopping; '
                                 'disabled by default (default: %(default)s)')
    validation.add_argument('--validate_only', action='store_true',
                            help='perform external validation with a pre-trained model')
    validation.add_argument('--bleu_script', type=str, default=None, metavar='PATH',
                            help='path to the external validation script (default: %(default)s); '
                                 'receives path of translation source file; must write a single score to STDOUT')
    validation.add_argument('--score_translations', action='store_true',
                            help='scores translations provided in a target file according to the learned model')

    display = parser.add_argument_group('display parameters')
    display.add_argument('--disp_freq', type=int, default=100, metavar='INT',
                         help='training metrics display frequency (default: %(default)s)')
    display.add_argument('--greedy_freq', type=int, default=1000, metavar='INT',
                         help='greedy sampling frequency (default: %(default)s)')
    display.add_argument('--sample_freq', type=int, default=0, metavar='INT',
                         help='weighted sampling frequency; disabled by default (default: %(default)s)')
    display.add_argument('--beam_freq', type=int, default=10000, metavar='INT',
                         help='beam search sampling frequency (default: %(default)s)')
    display.add_argument('--beam_size', type=int, default=4, metavar='INT',
                         help='size of the decoding beam (default: %(default)s)')

    translation = parser.add_argument_group('translation parameters')
    translation.add_argument('--translate_only', action='store_true',
                             help='translate a specified corpus using a pre-trained model')
    translation.add_argument('--translate_source_file', type=str, metavar='PATH',
                             help='corpus to be translated; must be pre-processed')
    translation.add_argument('--translate_target_file', type=str, metavar='PATH',
                             help='translation destination')
    translation.add_argument('--translate_with_beam_search', action='store_true',
                             help='translate using beam search')
    translation.add_argument('--length_normalization_alpha', type=float, default=0.6, metavar='FLOAT',
                             help='adjusts the severity of length penalty during beam decoding (default: %(default)s)')
    translation.add_argument('--no_normalize', action='store_true',
                             help='disable length normalization')
    translation.add_argument('--full_beam', action='store_true',
                             help='return all translation hypotheses within the beam')
    translation.add_argument('--translation_max_len', type=int, default=400, metavar='INT',
                             help='Maximum length of translation output sentence (default: %(default)s)')

    config = parser.parse_args()

    if not config.source_dataset:
        logging.error('--source_dataset is required')
        sys.exit(1)
    if not config.target_dataset:
        logging.error('--target_dataset is required')
        sys.exit(1)

    # Put check in place until factors are implemented
    if len(config.dictionaries) != 2:
        logging.error('exactly two dictionaries need to be provided')
        sys.exit(1)
    config.source_vocab = config.dictionaries[0]
    config.target_vocab = config.dictionaries[-1]

    # Embiggen the model
    if config.embiggen_model:
        config.embedding_size = 1024
        config.ffn_hidden_size = 4096
        config.hidden_size = 1024
        config.num_heads = 16
        config.dropout_embeddings = 0.3
        config.adam_beta2 = 0.998
        config.warmup_steps = 16000

    return config


if __name__ == "__main__":

    # IMPORTANT: Limit the number of reserved GPUs via 'export CUDA_VISIBLE_DEVICES $GPU_ID'

    # Assemble config
    config = parse_args()

    # Logging to file
    filemode = 'a' if config.reload else 'w'
    logging.basicConfig(filename=config.log_file, filemode=filemode, level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    if config.log_file is not None:
        # Logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    # Log the configuration when (re-)starting training/ validation/ translation
    logging.info('\nRUN CONFIGURATION')
    logging.info('=================')
    for key, val in config.__dict__.items():
        logging.info('{:s}: {}'.format(key, val))
    logging.info('=================\n')

    # Configure session
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = False

    # Filter out memory warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    with tf.Graph().as_default():
        if config.translate_only:
            # Translate a file
            if not config.translate_source_file:
                logging.error('--translate_source_file is required')
                sys.exit(1)
            if not config.translate_target_file:
                logging.error('--translate_target_file is required')
                sys.exit(1)
            translate(config, sess_config)
        elif config.validate_only:
            validate(config, sess_config)
        elif config.score_translations:
            translation_scorer(config, sess_config)
        else:
            train(config, sess_config)
