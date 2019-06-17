""" Build a neural machine translation model based on the transformer architecture and evaluate its hidden states. """

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import shuffle

import tensorflow as tf

from datetime import datetime
from collections import OrderedDict

from lexical_shortcuts.lexical_classifier import LexicalClassifier
from lexical_shortcuts.transformer_probed import Transformer as BaseTransformer
from lexical_shortcuts.lexical_shortcuts_transformer_probed import Transformer as LexicalShortcutsTransformer

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


def session_setup(config, sess, classifier, training=False, max_checkpoints=10):
    """ Prepares the model and auxiliary resources for operation. """
    to_init = list()
    # Exclude optimization variables to be loaded during inference (for greater model portability)
    nmt_to_load = None
    cls_vars = dict()

    global_vars = tf.global_variables()
    for var in global_vars:
        if 'classifier' in var.name:
            cls_vars[var.name.split(':')[0]] = var

    if not training:
        nmt_to_load = dict()
        for var in global_vars:
            if 'classifier' not in var.name:
                if 'optimization' not in var.name:  # remove in future
                    nmt_to_load[var.name.split(':')[0]] = var
                else:
                    to_init.append(var)

    # If a stand-alone model is called, variable names don't need to be mapped
    saver = tf.train.Saver(nmt_to_load, max_to_keep=max_checkpoints)
    cls_saver = tf.train.Saver(cls_vars, max_to_keep=max_checkpoints)
    reload_filename = None
    no_averaging = True

    if type(config.reload) == list and len(config.reload) > 1:
        reload_filename = average_checkpoints(nmt_to_load, config, sess)
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

    cls_reload_filename = None
    if config.cls_reload is not None:
        cls_reload_filename = config.cls_reload

    # Initialize a progress tracking object and restore its values, if possible
    progress = TrainingProgress()
    progress.bad_counter = 0
    progress.uidx = 0
    progress.eidx = 0
    progress.estop = False
    progress.validation_loss = OrderedDict()
    progress.validation_accuracy = OrderedDict()

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
            classifier.load_global_step(progress.uidx, sess)
        logging.info('Done!')

    if cls_reload_filename is not None:
        cls_saver.restore(sess, os.path.abspath(cls_reload_filename))
    else:
        cls_init_op = tf.variables_initializer(list(cls_vars.values()))
        sess.run(cls_init_op)

    logging.info('Finished setting up the model!')

    if training:
        return cls_saver, cls_reload_filename, saver, reload_filename, progress
    else:
        return cls_saver, cls_reload_filename, saver, reload_filename, progress


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
                                         sort_by_length=False,
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

    # Set up classifier
    vocab_size = source_vocab_size if config.probe_encoder else target_vocab_size
    classifier = LexicalClassifier(config, vocab_size)

    # Save model options
    config_as_dict = OrderedDict(sorted(vars(config).items()))
    json.dump(config_as_dict, open('{:s}.json'.format(config.save_to), 'w'), indent=2)

    # Initialize session
    sess = tf.Session(config=sess_config)
    if config.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root=None)
        sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    # Get validation and translation OPs
    if config.num_gpus >= 2:
        validation_ops = \
            get_parallel_ops(model, train_valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training', True)
        logging.info('[Parallel training, gradient delay == {:d}]'.format(config.gradient_delay))
    else:
        validation_ops = \
            get_single_ops(model, train_valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training', True)
        logging.info('[Single-device training, gradient delay == {:d}]'.format(config.gradient_delay))

    # Unpack validation and translation OPs
    _, batch_loss_op, _, _ = validation_ops
    cls_train_op, cls_batch_loss_op, cls_batch_accuracy_op, _ = classifier.train_model()

    logging.info('-' * 20)
    model_size = count_parameters()
    logging.info('Number of in-graph parameters (without activations): {:d}'.format(int(model_size)))
    logging.info('-' * 20)

    # Prepare model
    cls_saver, cls_checkpoint_path, saver, checkpoint_path, progress = \
        session_setup(config, sess, classifier, training=False, max_checkpoints=config.max_checkpoints)

    if checkpoint_path is not None:
        logging.info('NMT model restored from checkpoint {:s}'.format(checkpoint_path))

    if cls_checkpoint_path is not None:
        logging.info('Resuming classifier training from checkpoint {:s}'.format(cls_checkpoint_path))

    # Initialize iterator handles
    train_handle, valid_handle = sess.run([train_iterator.string_handle(), valid_iterator.string_handle()])

    # Initialize containers
    stored_inputs = list()
    stored_activations = list()
    classifier_global_step = 0
    stored_batches = 0
    cached_classifier_data = list()
    classifier_losses = list()
    classifier_accuracy = list()
    finished = False
    preprocessed = False
    classifier_turn = False
    early_stopped = False
    classifier_batch_size = 512
    save_path = config.save_to + '.classifier'
    num_sentences = 0

    logging.info('[BEGIN CLASSIFIER TRAINING]')
    logging.info('Current global step: {:d}'.format(progress.uidx))
    logging.info('-' * 20)

    for epoch_id in range(progress.eidx, config.max_epochs):
        # Check if training has been early stopped
        if progress.estop:
            break

        logging.info('Current training epoch: {:d}'.format(epoch_id))
        logging.info('-' * 20)

        # (Re-)initialize the training iterator
        sess.run(train_init_op)

        while True:

            # Training step
            if not classifier_turn and not finished:
                # Update flag
                preprocessed = False

                try:
                    # Define feed_dict
                    model_feed_dict = {iterator_handle: train_handle}

                    # Get model activations
                    _, source_inputs, target_inputs, enc_layer_outs, dec_layer_outs = \
                        sess.run([batch_loss_op, model.source_ids, model.target_ids_in, model.enc.layer_outputs,
                                  model.dec.layer_outputs], feed_dict=model_feed_dict)

                    # Store
                    probed_dict = enc_layer_outs if config.probe_encoder else dec_layer_outs
                    probed_inputs = source_inputs if config.probe_encoder else target_inputs
                    probed_activations = probed_dict['layer_{:d}'.format(config.probe_layer)][0]
                    stored_inputs.append(probed_inputs)
                    stored_activations.append(probed_activations)
                    stored_batches += 1
                    num_sentences += probed_inputs.shape[0]

                    # Report
                    if stored_batches % 20 == 0:
                        logging.info('Stored {:d} activations batches ({:d} sentences).'
                                     .format(stored_batches, num_sentences))

                    # Switch turns
                    if stored_batches >= 100:
                        classifier_turn = True

                except tf.errors.OutOfRangeError:
                    finished = True
                    classifier_turn = True

            else:
                if not preprocessed:
                    # Pre-process classifier inputs
                    sentence_activations = list()
                    sentence_inputs = list()

                    for batch_id, batch_activations in enumerate(stored_activations):
                        sentence_activations += \
                            np.split(batch_activations, indices_or_sections=batch_activations.shape[0], axis=0)
                        sentence_inputs += np.split(stored_inputs[batch_id],
                                                    indices_or_sections=stored_inputs[batch_id].shape[0], axis=0)

                    # Empty containers
                    stored_activations = list()
                    stored_inputs = list()

                    # Split into tokens
                    token_activations = [np.split(np.squeeze(s_a, axis=0), indices_or_sections=s_a.shape[1], axis=0)
                                         for s_a in sentence_activations]
                    token_inputs = [np.split(np.squeeze(s_i, axis=0), indices_or_sections=s_i.shape[1], axis=0)
                                    for s_i in sentence_inputs]

                    # Filter out padding
                    temp = list()
                    for sent_id, sent_act in enumerate(token_activations):
                        temp.append(sent_act[:len(token_inputs[sent_id])])
                    token_activations = temp

                    # Filter out <EOS> and <GO>
                    if config.probe_encoder:
                        for sent_id in range(len(sentence_activations)):
                            token_activations[sent_id] = token_activations[sent_id][:-1]
                            token_inputs[sent_id] = token_inputs[sent_id][:-1]
                    else:
                        for sent_id in range(len(sentence_activations)):
                            token_activations[sent_id] = token_activations[sent_id][1:]
                            token_inputs[sent_id] = token_inputs[sent_id][1:]

                    # Zip token activations with labels
                    zipped_sentences = list(zip(token_activations, token_inputs))
                    zipped_token = [list(zip(tpl[0], tpl[1])) for tpl in zipped_sentences]
                    cached_classifier_data = [item for sublist in zipped_token for item in sublist]
                    preprocessed = True

                    # Shuffle training data
                    shuffle(cached_classifier_data)

                else:
                    # Build-a-batch
                    try:
                        zipped_batch = cached_classifier_data[: classifier_batch_size]
                        cached_classifier_data = cached_classifier_data[classifier_batch_size:]
                    except IndexError:
                        zipped_batch = cached_classifier_data
                        cached_classifier_data = list()

                    # Make into source and target arrays
                    inputs_batch, labels_batch = zip(*zipped_batch)
                    inputs_batch = np.concatenate(inputs_batch, axis=0)
                    labels_batch = np.concatenate(labels_batch, axis=0)

                    # Define feed-dict
                    class_feed_dict = {classifier.inputs: inputs_batch,
                                       classifier.labels: labels_batch,
                                       classifier.training: True}

                    # Feed to the classifier
                    _, batch_loss, batch_accuracy, classifier_global_step = \
                        sess.run([cls_train_op, cls_batch_loss_op, cls_batch_accuracy_op, classifier.global_step],
                                 feed_dict=class_feed_dict)
                    classifier_losses.append(batch_loss)
                    classifier_accuracy.append(batch_accuracy)

                    # Report intermediate
                    if classifier_global_step % 100 == 0:
                        logging.info('-' * 20)
                        current_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
                        logging.info('{:s}[TRAIN] Epoch {:d} | Step {:d} | Loss/ batch {:4f} | Accuracy/ batch {:4f}'
                                     .format(current_time, epoch_id, classifier_global_step,
                                             np.mean(classifier_losses), np.mean(classifier_accuracy)))
                        logging.info('-' * 20)

                        classifier_losses = list()
                        classifier_accuracy = list()

                    # Report final
                    if len(cached_classifier_data) == 0:

                        logging.info('-' * 20)
                        current_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
                        logging.info('{:s}[TRAIN] Epoch {:d} | Step {:d} | Loss/ batch {:4f} | Accuracy/ batch {:4f}'
                                     .format(current_time, epoch_id, classifier_global_step,
                                             np.mean(classifier_losses), np.mean(classifier_accuracy)))
                        logging.info('-' * 20)

                        # Update flags, containers
                        classifier_turn = False
                        stored_batches = 0
                        classifier_losses = list()
                        classifier_accuracy = list()

                        if finished:
                            break

                    # Validation step
                    if config.valid_freq and classifier_global_step % config.valid_freq == 0:
                        logging.info('[BEGIN CLASSIFIER VALIDATION]')
                        logging.info('-' * 20)
                        # (Re-)initialize the validation iterator
                        sess.run(valid_init_op)
                        validation_ops = [batch_loss_op, cls_batch_loss_op, cls_batch_accuracy_op]
                        handles = [iterator_handle, valid_handle]

                        # Get validation loss
                        validation_loss, validation_accuracy = \
                            validation_loop(sess, model, classifier, validation_ops, handles)

                        if len(progress.validation_accuracy) == 0 or \
                                validation_accuracy > max(list(progress.validation_accuracy.values())):
                            progress.validation_accuracy[int(classifier_global_step)] = validation_accuracy
                            progress.validation_loss[int(classifier_global_step)] = validation_loss

                            # Save model checkpoint in case validation accuracy has improved
                            cls_saver.save(sess,
                                           save_path='{:s}-best_classifier_validation_accuracy'.format(config.save_to))
                            logging.info(
                                '[CHECKPOINT] Saved a best-accuracy-loss model checkpoint to {:s}.'.format(
                                    config.save_to))
                            progress_path = '{:s}-best_classifier_validation_accuracy.progress.json'.format(
                                config.save_to)
                            progress.save_to_json(progress_path)
                            logging.info('-' * 20)
                            progress.bad_counter = 0
                        else:
                            # Track validation accuracy
                            progress.validation_accuracy[int(classifier_global_step)] = validation_accuracy
                            progress.validation_loss[int(classifier_global_step)] = validation_loss

                            # Check for early-stopping
                            progress.bad_counter += 1
                            if progress.bad_counter >= config.patience > 0:
                                # Execute early stopping of the training
                                logging.info(
                                    'No improvement observed on the validation set for {:d} steps. Early stop!'
                                        .format(progress.bad_counter))
                                progress.estop = True
                                early_stopped = True
                                break

                # Save model parameters
                if config.save_freq and classifier_global_step % config.save_freq == 0 and classifier_global_step > 0:
                    cls_saver.save(sess, save_path=save_path, global_step=classifier_global_step)
                    logging.info(
                        '[CHECKPOINT] Saved a scheduled classifier checkpoint to {:s}.'.format(config.save_to))
                    logging.info('-' * 20)
                    progress_path = '{:s}-{:d}.classifier_progress.json'.format(config.save_to, classifier_global_step)
                    progress.save_to_json(progress_path)

                if config.max_updates and classifier_global_step % config.max_updates == 0 \
                        and classifier_global_step > 0:
                    logging.info('Maximum number of updates reached!')
                    saver.save(sess, save_path=save_path, global_step=progress.uidx)
                    logging.info('[CHECKPOINT] Saved the training-final classifier checkpoint to {:s}.'
                                 .format(config.save_to))
                    logging.info('-' * 20)
                    progress.estop = True
                    progress_path = '{:s}-{:d}.classifier_progress.json'.format(config.save_to, progress.uidx)
                    progress.save_to_json(progress_path)
                    break

        if not early_stopped:
            logging.info('Epoch {:d} concluded'.format(epoch_id))
            # Update the persistent global step tracker
            progress.uidx = int(classifier_global_step)
            # Update the persistent epoch tracker
            progress.eidx += 1

    # Close active session
    sess.close()


def validation_loop(sess, model, classifier, ops, handles):
    """ Iterates over the validation data, calculating a trained model's cross-entropy. """
    # Unpack OPs
    batch_loss_op, cls_batch_loss_op, cls_batch_accuracy_op = ops

    # Initialize metrics
    stored_inputs = list()
    stored_activations = list()
    stored_batches = 0
    seen_batches = 0
    classifier_batch_size = 512
    cached_classifier_data = list()
    interval_validation_losses = list()
    validation_losses = list()
    interval_validation_accuracy = list()
    validation_accuracy = list()
    finished = False
    preprocessed = False
    classifier_turn = False
    num_sentences = 0

    # Unpack iterator variables
    if handles is not None:
        handle, valid_handle = handles
        model_feed_dict = {handle: valid_handle,
                           model.training: False}
    else:
        model_feed_dict = {model.training: False}

    logging.info('Estimating validation loss ... ')
    while True:
        # Run a forward pass through the model and classifier
        # Training step
        if not classifier_turn and not finished:
            # Update flag
            preprocessed = False

            try:
                # Get model activations
                _, source_inputs, target_inputs, enc_layer_outs, dec_layer_outs = \
                    sess.run([batch_loss_op, model.source_ids, model.target_ids_in, model.enc.layer_outputs,
                              model.dec.layer_outputs], feed_dict=model_feed_dict)

                # Store
                probed_dict = enc_layer_outs if config.probe_encoder else dec_layer_outs
                probed_inputs = source_inputs if config.probe_encoder else target_inputs
                probed_activations = probed_dict['layer_{:d}'.format(config.probe_layer)][0]
                stored_inputs.append(probed_inputs)
                stored_activations.append(probed_activations)
                stored_batches += 1
                num_sentences += probed_inputs.shape[0]

                # Report
                if stored_batches % 20 == 0:
                    logging.info('Stored {:d} activations batches ({:d} sentences).'
                                 .format(stored_batches, num_sentences))

                # Switch turns
                if stored_batches >= 100:
                    classifier_turn = True

            except tf.errors.OutOfRangeError:
                finished = True

        else:
            if not preprocessed:
                # Pre-process classifier inputs
                sentence_activations = list()
                sentence_inputs = list()

                for batch_id, batch_activations in enumerate(stored_activations):
                    sentence_activations += \
                        np.split(batch_activations, indices_or_sections=batch_activations.shape[0], axis=0)
                    sentence_inputs += np.split(stored_inputs[batch_id],
                                                indices_or_sections=stored_inputs[batch_id].shape[0], axis=0)

                # Empty containers
                stored_activations = list()
                stored_inputs = list()

                # Split into tokens
                token_activations = [np.split(np.squeeze(s_a, axis=0), indices_or_sections=s_a.shape[1], axis=0)
                                     for s_a in sentence_activations]
                token_inputs = [np.split(np.squeeze(s_i, axis=0), indices_or_sections=s_i.shape[1], axis=0)
                                for s_i in sentence_inputs]

                # Filter out padding
                temp = list()
                for sent_id, sent_act in enumerate(token_activations):
                    temp.append(sent_act[:len(token_inputs[sent_id])])
                token_activations = temp

                # Filter out <EOS>
                if not config.probe_encoder:
                    for sent_id in range(len(sentence_activations)):
                        token_activations[sent_id] = token_activations[sent_id][:-1]
                        token_inputs[sent_id] = token_inputs[sent_id][:-1]

                # Zip token activations with labels
                zipped_sentences = list(zip(token_activations, token_inputs))
                zipped_token = [list(zip(tpl[0], tpl[1])) for tpl in zipped_sentences]
                cached_classifier_data = [item for sublist in zipped_token for item in sublist]
                preprocessed = True

                seen_batches = 0

            else:
                # Build-a-batch
                try:
                    zipped_batch = cached_classifier_data[: classifier_batch_size]
                    cached_classifier_data = cached_classifier_data[classifier_batch_size:]
                except IndexError:
                    zipped_batch = cached_classifier_data
                    cached_classifier_data = list()

                # Make into source and target arrays
                inputs_batch, labels_batch = zip(*zipped_batch)
                inputs_batch = np.concatenate(inputs_batch, axis=0)
                labels_batch = np.concatenate(labels_batch, axis=0)

                # Define feed-dict
                class_feed_dict = {classifier.inputs: inputs_batch,
                                   classifier.labels: labels_batch,
                                   classifier.training: False}

                # Feed to the classifier
                batch_loss, batch_accuracy, classifier_global_step = \
                    sess.run([cls_batch_loss_op, cls_batch_accuracy_op, classifier.global_step],
                             feed_dict=class_feed_dict)
                interval_validation_losses.append(batch_loss)
                interval_validation_accuracy.append(batch_accuracy)

                seen_batches += 1

                # Report
                if seen_batches % 20 == 0:
                    logging.info('Evaluated {:d} batches'.format(seen_batches))

                # Report
                if len(cached_classifier_data) == 0:

                    # Update flag
                    classifier_turn = False
                    stored_batches = 0

                    logging.info('-' * 20)
                    current_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
                    logging.info('{:s}[VALID] Step {:d} | Loss/ batch {:4f} | Accuracy/ batch {:4f}'
                                 .format(current_time, classifier_global_step, np.mean(interval_validation_losses),
                                         np.mean(interval_validation_accuracy)))

                    logging.info('-' * 20)
                    validation_losses += interval_validation_losses
                    validation_accuracy += interval_validation_accuracy

                    interval_validation_losses = list()
                    interval_validation_accuracy = list()

                    if finished:
                        break

    # Report
    mean_valid_loss = float(np.mean(validation_losses))
    mean_valid_accuracy = float(np.mean(validation_accuracy))

    current_time = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
    logging.info('-' * 20)
    logging.info('{:s}[VALID] Avg. loss: {:.4f} | Avg. accuracy {:.4f}| Sentence total {:d}'
                 .format(current_time, mean_valid_loss, mean_valid_accuracy, num_sentences))

    return mean_valid_loss, mean_valid_accuracy


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

    # Set up classifier
    vocab_size = source_vocab_size if config.probe_encoder else target_vocab_size
    classifier = LexicalClassifier(config, vocab_size)

    # Save model options
    config_as_dict = OrderedDict(sorted(vars(config).items()))
    json.dump(config_as_dict, open('{:s}.json'.format(config.save_to), 'w'), indent=2)

    # Initialize session
    sess = tf.Session(config=sess_config)
    if config.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root=None)
        sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    # Get validation and translation OPs
    if config.num_gpus >= 2:
        validation_ops = \
            get_parallel_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training', True)
        logging.info('[Parallel training, gradient delay == {:d}]'.format(config.gradient_delay))
    else:
        validation_ops = \
            get_single_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training', True)
        logging.info('[Single-device training, gradient delay == {:d}]'.format(config.gradient_delay))

    # Unpack validation and translation OPs
    _, batch_loss_op, _, _ = validation_ops
    _, cls_batch_loss_op, cls_batch_accuracy_op, _ = classifier.train_model()

    logging.info('-' * 20)
    model_size = count_parameters()
    logging.info('Number of in-graph parameters (without activations): {:d}'.format(int(model_size)))
    logging.info('-' * 20)

    # Prepare model
    cls_saver, cls_checkpoint_path, saver, checkpoint_path, progress = \
        session_setup(config, sess, classifier, training=False, max_checkpoints=config.max_checkpoints)

    logging.info('-' * 20)
    if checkpoint_path is not None:
        logging.info('NMT model restored from checkpoint {:s}'.format(checkpoint_path))
    else:
        logging.info('No checkpoint to initialize the NMT model from could be found. Exiting.')
        sys.exit(1)

    if cls_checkpoint_path is not None:
        logging.info('Classifier restored from checkpoint {:s}'.format(cls_checkpoint_path))
    else:
        logging.info('No checkpoint to initialize the classifier from could be found. Exiting.')
        sys.exit(1)

    logging.info('-' * 20)
    logging.info('Performing validation on corpus {:s}'.format(config.valid_target_dataset, model.name))
    logging.info('[BEGIN VALIDATION]')
    logging.info('-' * 20)

    # Validate
    sess.run(valid_iterator.initializer)
    validation_ops = [batch_loss_op, cls_batch_loss_op, cls_batch_accuracy_op]

    # Get validation loss
    _, _ = validation_loop(sess, model, classifier, validation_ops, None)


def pos_freq_match(config, sess_config):
    """ Helper function for checking the fine-grained classification errors of the lexical predictor. """

    assert config.reload is not None, \
        'Model path is not specified. Set path to model checkpoint using the --reload flag.'

    assert config.token_batch_size == 0 and config.sentence_batch_size == 1, \
        'POS/ FREQ matching mode requires single sentences as input.'

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

    # Set up classifier
    vocab_size = source_vocab_size if config.probe_encoder else target_vocab_size
    classifier = LexicalClassifier(config, vocab_size)

    # Save model options
    config_as_dict = OrderedDict(sorted(vars(config).items()))
    json.dump(config_as_dict, open('{:s}.json'.format(config.save_to), 'w'), indent=2)

    # Initialize session
    sess = tf.Session(config=sess_config)
    if config.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root=None)
        sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    # Get validation and translation OPs
    validation_ops = get_single_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training', True)
    logging.info('[Single-device training, gradient delay == {:d}]'.format(config.gradient_delay))

    # Unpack validation and translation OPs
    _, batch_loss_op, _, _ = validation_ops
    _, cls_batch_loss_op, cls_batch_accuracy_op, cls_correct_predictions = classifier.train_model()

    logging.info('-' * 20)
    model_size = count_parameters()
    logging.info('Number of in-graph parameters (without activations): {:d}'.format(int(model_size)))
    logging.info('-' * 20)

    # Prepare model
    cls_saver, cls_checkpoint_path, saver, checkpoint_path, progress = \
        session_setup(config, sess, classifier, training=False, max_checkpoints=config.max_checkpoints)

    if not os.path.exists(config.cls_pickle_dir):
        os.makedirs(config.cls_pickle_dir)

    eval_lang = config.pos_reference.split('.')[-2]

    logging.info('-' * 20)
    if checkpoint_path is not None:
        logging.info('NMT model restored from checkpoint {:s}'.format(checkpoint_path))
    else:
        logging.info('No checkpoint to initialize the NMT model from could be found. Exiting.')
        sys.exit(1)

    if cls_checkpoint_path is not None:
        logging.info('Classifier restored from checkpoint {:s}'.format(cls_checkpoint_path))
    else:
        logging.info('No checkpoint to initialize the classifier from could be found. Exiting.')
        sys.exit(1)

    logging.info('-' * 20)
    logging.info('Performing evaluation on corpus {:s}'.format(config.valid_target_dataset, model.name))
    logging.info('[BEGIN EVALUATION]')
    logging.info('-' * 20)
    sess.run(valid_iterator.initializer)

    # Initialize metrics
    stored_inputs = list()
    stored_activations = list()
    stored_batches = 0
    seen_batches = 0
    cached_classifier_data = list()
    finished = False
    preprocessed = False
    classifier_turn = False
    num_sentences = 0

    model_feed_dict = {model.training: False}

    # Get POS & frequency tag references
    with open(config.pos_reference, 'r') as pos_ref:
        pos_lines = pos_ref.readlines()
    with open(config.freq_reference, 'r') as freq_ref:
        feq_lines = freq_ref.readlines()

    # Store match information
    pos_matches = OrderedDict([('NOUN', list()), ('VERB', list()), ('ADJ/ADV', list()), ('OTHER', list())])
    freq_matches = OrderedDict([(bucket_id, list()) for bucket_id in range(1, 11)])

    logging.info('Evaluating matches ... ')
    while True:
        # Run a forward pass through the model and classifier
        # Training step
        if not classifier_turn and not finished:
            # Update flag
            preprocessed = False

            try:
                # Get model activations
                _, source_inputs, target_inputs, enc_layer_outs, dec_layer_outs = \
                    sess.run([batch_loss_op, model.source_ids, model.target_ids_in, model.enc.layer_outputs,
                              model.dec.layer_outputs], feed_dict=model_feed_dict)

                # Store
                probed_dict = enc_layer_outs if config.probe_encoder else dec_layer_outs
                probed_inputs = source_inputs if config.probe_encoder else target_inputs
                probed_activations = probed_dict['layer_{:d}'.format(config.probe_layer)][0]
                stored_inputs.append(probed_inputs)
                stored_activations.append(probed_activations)
                stored_batches += 1
                num_sentences += probed_inputs.shape[0]

                # Switch turns
                if stored_batches == 1:
                    classifier_turn = True

            except tf.errors.OutOfRangeError:
                finished = True

        else:

            if finished:
                break

            if not preprocessed:
                # Pre-process classifier inputs
                sentence_activations = list()
                sentence_inputs = list()

                for batch_id, batch_activations in enumerate(stored_activations):
                    sentence_activations += \
                        np.split(batch_activations, indices_or_sections=batch_activations.shape[0], axis=0)
                    sentence_inputs += np.split(stored_inputs[batch_id],
                                                indices_or_sections=stored_inputs[batch_id].shape[0], axis=0)

                # Empty containers
                stored_activations = list()
                stored_inputs = list()

                # Split into tokens
                token_activations = [np.split(np.squeeze(s_a, axis=0), indices_or_sections=s_a.shape[1], axis=0)
                                     for s_a in sentence_activations]
                token_inputs = [np.split(np.squeeze(s_i, axis=0), indices_or_sections=s_i.shape[1], axis=0)
                                for s_i in sentence_inputs]

                # Filter out padding
                temp = list()
                for sent_id, sent_act in enumerate(token_activations):
                    temp.append(sent_act[:len(token_inputs[sent_id])])
                token_activations = temp

                # Filter out <EOS> and <GO>
                if config.probe_encoder:
                    for sent_id in range(len(sentence_activations)):
                        token_activations[sent_id] = token_activations[sent_id][:-1]
                        token_inputs[sent_id] = token_inputs[sent_id][:-1]
                else:
                    for sent_id in range(len(sentence_activations)):
                        token_activations[sent_id] = token_activations[sent_id][1:]
                        token_inputs[sent_id] = token_inputs[sent_id][1:]

                # Zip token activations with labels
                zipped_sentences = list(zip(token_activations, token_inputs))
                zipped_token = [list(zip(tpl[0], tpl[1])) for tpl in zipped_sentences]
                cached_classifier_data = [item for sublist in zipped_token for item in sublist]
                preprocessed = True

            else:
                # Build-a-batch
                zipped_batch = cached_classifier_data
                cached_classifier_data = list()

                # Make into source and target arrays
                inputs_batch, labels_batch = zip(*zipped_batch)
                inputs_batch = np.concatenate(inputs_batch, axis=0)
                labels_batch = np.concatenate(labels_batch, axis=0)

                # Define feed-dict
                class_feed_dict = {classifier.inputs: inputs_batch,
                                   classifier.labels: labels_batch,
                                   classifier.training: False}

                # Feed to the classifier
                _, batch_accuracy, correct_predictions = \
                    sess.run([cls_batch_loss_op, cls_batch_accuracy_op, cls_correct_predictions],
                             feed_dict=class_feed_dict)

                # Check POS/ FREQ match
                # correct predictions has shape=[batch_size,], i.e. [sent_len] when using sent_bach_size == 1
                curr_pos_line = pos_lines[seen_batches].strip().split('||')[:-1]
                curr_freq_line = feq_lines[seen_batches].strip().split('||')[:-1]
                # Extract POS/ FREQ tags from lines
                sw_tokens = [pos_tpl.strip().split(' ')[0] for pos_tpl in curr_pos_line]
                sw_pos_tags = [pos_tpl.strip().split(' ')[1] for pos_tpl in curr_pos_line]
                freq_tags = [freq_tpl.strip().split(' ')[1] for freq_tpl in curr_freq_line]
                cls_hits = correct_predictions.tolist()

                assert len(sw_pos_tags) == len(freq_tags) == len(cls_hits), \
                    'Label / prediction mismatch: POS {:d}, FREQ {:d}, HITS {:d}' \
                        .format(len(sw_pos_tags), len(freq_tags), len(cls_hits))

                # Combine sub-word POS-predictions for word-level evaluation
                pos_tags = list()
                pos_hits = list()
                open_sw = False
                word_pos = None
                word_hits = None

                for swt_id, swt in enumerate(sw_tokens):

                    if swt.endswith('@@'):
                        open_sw = True

                    if open_sw:
                        word_pos = sw_pos_tags[swt_id]
                        if word_hits is not None:
                            word_hits.append(cls_hits[swt_id])
                        else:
                            word_hits = [cls_hits[swt_id]]

                    if not swt.endswith('@@'):
                        open_sw = False

                        if word_hits is not None:
                            pos_tags.append(word_pos)
                            pos_hits.append(np.max(word_hits))
                            # Reset
                            word_hits = None
                        else:
                            pos_tags.append(sw_pos_tags[swt_id])
                            pos_hits.append(cls_hits[swt_id])

                # Document and store correct predictions
                # ENGLISH POS-tag super-categories
                # closed = [ls,:, in, dt, cd, wp, to, sent, cc, pos, pp, md, ", ,, ``, pp$, wdt, rp, wrb, sym, pdt,
                #     (, ex, ), wp$, UH, FW, $, #]
                # nouns = [nn, np, nns, nps]; if starts with 'N'
                # verbs = [vbd, vbn, vbz, vbp, vb, vbg]; if starts with 'V'
                # adjectives & adverbs = [jj, rb, jjr, jjs, rbr, rbs]; if starts with 'J' or RB'

                # GERMAN POS-tag super-categories
                # nouns = if starts with 'N'
                # verbs = if starts with 'V'
                # adjectives & adverbs = if starts with 'ADJ' or is 'ADV'
                # closed = everything else

                # RUSSIAN POS-tag super-categories (should be roughly correct)
                # nouns = if starts with 'N'
                # verbs = if starts with 'V'
                # adjectives & adverbs = if starts with 'A'
                # closed = everything else

                for tag_id, pos_tag in enumerate(pos_tags):
                    # Update dicts
                    if pos_tag.startswith('N'):
                        super_tag = 'NOUN'
                    elif pos_tag.startswith('V'):
                        super_tag = 'VERB'
                    elif pos_tag.startswith('A') and (eval_lang == 'de' or eval_lang == 'ru'):
                        if eval_lang == 'ru':
                            super_tag = 'ADJ/ADV'
                        else:
                            if pos_tag.startswith('ADJ') or pos_tag == 'ADV':
                                super_tag = 'ADJ/ADV'
                            else:
                                super_tag = 'OTHER'
                    elif (pos_tag.startswith('J') or pos_tag.startswith('RB')) and eval_lang == 'en':
                        super_tag = 'ADJ/ADV'
                    else:
                        super_tag = 'OTHER'

                    pos_matches[super_tag].append(pos_hits[tag_id])

                for tag_id, freq_tag in enumerate(freq_tags):
                    freq_tag = int(freq_tag)
                    if freq_tag not in freq_matches.keys():
                        freq_matches[freq_tag] = [cls_hits[tag_id]]
                    else:
                        freq_matches[freq_tag].append(cls_hits[tag_id])

                seen_batches += 1

                # Report
                if seen_batches % 100 == 0:
                    logging.info('Evaluated {:d} sentences'.format(seen_batches))

                if len(cached_classifier_data) == 0:
                    # Update flag
                    classifier_turn = False
                    stored_batches = 0

    # Pickle
    tracked_model = 'encoder' if config.probe_encoder else 'decoder'
    with open(os.path.join(config.cls_pickle_dir,
                           '{:s}_{:d}.pkl'.format(tracked_model, config.probe_layer)), 'wb') as pkl_in:
        pickle.dump([pos_matches, freq_matches], pkl_in)

    # Report
    # Compute accuracy for each POS tag/ frequency bin
    logging.info('-' * 20)
    logging.info('\nPOS-BASED CLASSIFICATION ACCURACY')
    for super_tag in pos_matches.keys():
        logging.info('{:s} : {:.8f}'.format(super_tag, np.mean(pos_matches[super_tag])))
    logging.info('-' * 20)
    logging.info('\nFREQUENCY-BASED CLASSIFICATION ACCURACY')
    for freq_tag in freq_matches.keys():
        logging.info('{:d} : {:.8f}'.format(freq_tag, np.mean(freq_matches[freq_tag])))


def heatmap_from_pickle(config):
    """ Generates a heatmap from classifier predictions. """
    cls_pickles = os.listdir(config.cls_pickle_dir)
    # Separate
    enc_pickles = [(cls_p, int(cls_p.split('_')[-1].split('.')[0])) for cls_p in cls_pickles if 'encoder' in cls_p]
    dec_pickles = [(cls_p, int(cls_p.split('_')[-1].split('.')[0])) for cls_p in cls_pickles if 'decoder' in cls_p]
    # Sort
    enc_pickles.sort(key=lambda x: x[1])
    dec_pickles.sort(key=lambda x: x[1])
    # Drop embedding layer results
    enc_pickles = enc_pickles[1:]
    dec_pickles = dec_pickles[1:]
    # Make heat-maps
    for sub_net in ['encoder', 'decoder']:
        plot_save_dir = os.path.join('/'.join(config.cls_pickle_dir.split('/')[: -1]), '{:s}_plots'.format(sub_net))
        if not os.path.exists(plot_save_dir):
            os.makedirs(plot_save_dir)
        pkl_list = enc_pickles if sub_net == 'encoder' else dec_pickles
        pos_match_matrix = list()
        freq_match_matrix = list()
        layer_ids = ['layer {:d}'.format(l_id) for l_id in range(1, config.num_encoder_layers + 1)]
        pos_categories = None
        freq_categories = None
        freq_category_labels = None

        for layer_id, pkl in enumerate(pkl_list):
            with open(os.path.join(config.cls_pickle_dir, pkl[0]), 'rb') as pkl_out:
                pos_match_dict, freq_match_dict = pickle.load(pkl_out)
                pos_categories = list(pos_match_dict.keys()) if pos_categories is None else pos_categories
                freq_categories = list(freq_match_dict.keys()) if freq_categories is None else freq_categories
                if freq_categories is not None:
                    freq_category_labels = ['bin {:d}'.format(c) for c in freq_categories]
                pos_matches = [np.mean(pos_match_dict[cat_key]) for cat_key in pos_categories]
                freq_matches = [np.mean(freq_match_dict[cat_key]) for cat_key in freq_categories]
                pos_match_matrix.append(pos_matches)
                freq_match_matrix.append(freq_matches)

        pos_match_matrix = np.stack(pos_match_matrix, axis=0)
        freq_match_matrix = np.stack(freq_match_matrix, axis=0)

        pos_df = pd.DataFrame(pos_match_matrix,
                              index=layer_ids,
                              columns=pos_categories)

        plt.figure(figsize=(6, 4))

        sns.heatmap(pos_df, cmap='coolwarm', linewidths=0.25, vmin=0.0, vmax=1.0, xticklabels=True, yticklabels=True,
                    annot=True, square=True, fmt='.3f')
        plt.yticks(rotation=0)
        plot_path = os.path.join(plot_save_dir, '{:s}_pos_match.png'.format(sub_net))
        plt.savefig(plot_path, dpi='figure', pad_inches=1, bbox_inches='tight')
        plt.clf()

        plt.figure(figsize=(8, 8))

        freq_df = pd.DataFrame(freq_match_matrix,
                               index=layer_ids,
                               columns=freq_category_labels)

        sns.heatmap(
            freq_df, cmap='coolwarm', linewidths=0.25, vmin=0.0, vmax=1.0, xticklabels=True, yticklabels=True,
            annot=True, square=True, fmt='.3f', cbar_kws={'shrink': .4})
        plt.yticks(rotation=0)
        plot_path = os.path.join(plot_save_dir, '{:s}_freq_match.png'.format(sub_net))
        plt.savefig(plot_path, dpi='figure', pad_inches=1, bbox_inches='tight')
        plt.clf()

    logging.info('Done!')


def gate_evaluation(config, sess_config):
    """ Helper function for checking the fine-grained classification errors of the lexical predictor. """

    def hoyer_sparseness(arr):
        """ Estimates the sparseness of an array using the 'Hoyer' measure:
        https://math.stackexchange.com/questions/117860/how-to-define-sparseness-of-a-vector """
        return (np.sqrt(arr.shape[0]) - (np.sum(arr) / np.sqrt(np.sum(arr ** 2)))) / (np.sqrt(arr.shape[0]) - 1)

    assert config.reload is not None, \
        'Model path is not specified. Set path to model checkpoint using the --reload flag.'

    assert config.token_batch_size == 0 and config.sentence_batch_size == 1, \
        'POS/ FREQ matching mode requires single sentences as input.'

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

    # Set up classifier
    vocab_size = source_vocab_size if config.probe_encoder else target_vocab_size
    classifier = LexicalClassifier(config, vocab_size)

    # Save model options
    config_as_dict = OrderedDict(sorted(vars(config).items()))
    json.dump(config_as_dict, open('{:s}.json'.format(config.save_to), 'w'), indent=2)

    # Initialize session
    sess = tf.Session(config=sess_config)
    if config.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root=None)
        sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    # Get validation and translation OPs
    validation_ops = get_single_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training', True)
    logging.info('[Single-device training, gradient delay == {:d}]'.format(config.gradient_delay))

    # Unpack validation and translation OPs
    _, batch_loss_op, _, _ = validation_ops
    _, cls_batch_loss_op, _, cls_correct_predictions = classifier.train_model()

    # Prepare model
    cls_saver, cls_checkpoint_path, saver, checkpoint_path, progress = \
        session_setup(config, sess, classifier, training=False, max_checkpoints=config.max_checkpoints)

    # Plot directory
    tracked_model = 'encoder' if config.probe_encoder else 'decoder'
    plot_save_dir = os.path.join('/'.join(config.save_to.split('/')[:-1]), '{:s}_gate_plots'.format(tracked_model))
    if not os.path.exists(plot_save_dir):
        os.makedirs(plot_save_dir)

    eval_lang = config.pos_reference.split('.')[-2]

    logging.info('-' * 20)
    if checkpoint_path is not None:
        logging.info('NMT model restored from checkpoint {:s}'.format(checkpoint_path))
    else:
        logging.info('No checkpoint to initialize the NMT model from could be found. Exiting.')
        sys.exit(1)

    logging.info('-' * 20)
    logging.info('Performing evaluation on corpus {:s}'.format(config.valid_target_dataset, model.name))
    logging.info('[BEGIN EVALUATION]')
    logging.info('-' * 20)
    sess.run(valid_iterator.initializer)

    feed_dict = {model.training: False}

    # Get POS & frequency tag references
    with open(config.pos_reference, 'r') as pos_ref:
        pos_lines = pos_ref.readlines()
    with open(config.freq_reference, 'r') as freq_ref:
        feq_lines = freq_ref.readlines()

    seen_batches = 0

    # Store match information
    assert config.num_encoder_layers == config.num_decoder_layers, \
        'Number of encoder and decoder layers must be identical.'  # lazy hack
    all_keys_pos_matches = \
        [OrderedDict([('NOUN', list()), ('VERB', list()), ('ADJ/ADV', list()), ('OTHER', list())])
         for _ in range(config.num_encoder_layers)]
    all_keys_freq_matches = \
        [OrderedDict([(bucket_id, list()) for bucket_id in range(1, 11)]) for _ in range(config.num_encoder_layers)]
    all_keys_pos_sparseness = \
        [OrderedDict([('NOUN', list()), ('VERB', list()), ('ADJ/ADV', list()), ('OTHER', list())])
         for _ in range(config.num_encoder_layers)]
    all_keys_freq_sparseness = \
        [OrderedDict([(bucket_id, list()) for bucket_id in range(1, 11)]) for _ in range(config.num_encoder_layers)]

    all_values_pos_matches = \
        [OrderedDict([('NOUN', list()), ('VERB', list()), ('ADJ/ADV', list()), ('OTHER', list())])
         for _ in range(config.num_encoder_layers)]
    all_values_freq_matches = \
        [OrderedDict([(bucket_id, list()) for bucket_id in range(1, 11)]) for _ in range(config.num_encoder_layers)]
    all_values_pos_sparseness = \
        [OrderedDict([('NOUN', list()), ('VERB', list()), ('ADJ/ADV', list()), ('OTHER', list())])
         for _ in range(config.num_encoder_layers)]
    all_values_freq_sparsness = \
        [OrderedDict([(bucket_id, list()) for bucket_id in range(1, 11)]) for _ in range(config.num_encoder_layers)]

    logging.info('Evaluating gate activations ... ')
    while True:
        # Run a forward pass through the model and collect activations
        try:
            # Get model activations
            _, gate_tracker = sess.run([batch_loss_op, model.gate_tracker], feed_dict=feed_dict)

            # Check POS/ FREQ -specific activations
            curr_pos_line = pos_lines[seen_batches].strip().split('||')[:-1]
            curr_freq_line = feq_lines[seen_batches].strip().split('||')[:-1]

            # Extract POS/ FREQ tags from lines
            sw_tokens = [pos_tpl.strip().split(' ')[0] for pos_tpl in curr_pos_line]
            sw_pos_tags = [pos_tpl.strip().split(' ')[1] for pos_tpl in curr_pos_line]
            freq_tags = [freq_tpl.strip().split(' ')[1] for freq_tpl in curr_freq_line]

            # Aggregate activation values
            for layer_id in range(config.num_encoder_layers):
                keys_pos_matches = all_keys_pos_matches[layer_id]
                values_pos_matches = all_values_pos_matches[layer_id]

                keys_freq_matches = all_keys_freq_matches[layer_id]
                values_freq_matches = all_values_freq_matches[layer_id]

                keys_pos_sparseness = all_keys_pos_sparseness[layer_id]
                values_pos_sparseness = all_values_pos_sparseness[layer_id]

                keys_freq_sparseness = all_keys_freq_sparseness[layer_id]
                values_freq_sparseness = all_values_freq_sparsness[layer_id]

                keys_gate_activations_features = \
                    gate_tracker['{:s}_layer_{:d}'.format(tracked_model, layer_id + 1)]['lexical_gate_keys']
                values_gate_activations_features = \
                    gate_tracker['{:s}_layer_{:d}'.format(tracked_model, layer_id + 1)]['lexical_gate_values']

                # Remove special tokens
                keys_gate_activations_features = keys_gate_activations_features[:, :-1, :] \
                    if config.probe_encoder else keys_gate_activations_features[:, 1:, :]
                values_gate_activations_features = values_gate_activations_features[:, : -1, :] \
                    if config.probe_encoder else values_gate_activations_features[:, 1:, :]

                # Reduce features
                keys_gate_activations = \
                    np.squeeze(np.mean(keys_gate_activations_features, axis=-1), axis=0).tolist()
                values_gate_activations = \
                    np.squeeze(np.mean(values_gate_activations_features, axis=-1), axis=0).tolist()

                assert len(sw_pos_tags) == len(freq_tags) == len(keys_gate_activations), \
                    'Label / prediction mismatch: POS {:d}, FREQ {:d}, HITS {:d}' \
                        .format(len(sw_pos_tags), len(freq_tags), len(keys_gate_activations))

                # Estimate activation sparseness
                keys_gate_activations_token_features = \
                    [np.squeeze(kga, axis=0) for kga in
                     np.split(np.squeeze(keys_gate_activations_features, axis=0),
                              indices_or_sections=keys_gate_activations_features.shape[1], axis=0)]
                values_gate_activations_token_features = \
                    [np.squeeze(vga, axis=0) for vga in
                     np.split(np.squeeze(values_gate_activations_features, axis=0),
                              indices_or_sections=values_gate_activations_features.shape[1], axis=0)]

                # Combine subword POS-predictions for word-level evaluation
                pos_tags = list()
                pos_keys_gate_activations = list()
                pos_values_gate_activations = list()
                pos_keys_gate_activations_features = list()
                pos_values_gate_activations_features = list()
                open_sw = False
                word_pos = None
                word_keys_gate_activations = None
                word_values_gate_activations = None
                word_keys_gate_activations_features = None
                word_values_gate_activations_features = None
                for swt_id, swt in enumerate(sw_tokens):

                    if swt.endswith('@@'):
                        open_sw = True

                    if open_sw:
                        word_pos = sw_pos_tags[swt_id]
                        if word_keys_gate_activations is not None:
                            word_keys_gate_activations.append(keys_gate_activations[swt_id])
                            word_values_gate_activations.append(values_gate_activations[swt_id])

                            word_keys_gate_activations_features.append(
                                hoyer_sparseness(keys_gate_activations_token_features[swt_id]))
                            word_values_gate_activations_features.append(
                                hoyer_sparseness(values_gate_activations_token_features[swt_id]))

                        else:
                            word_keys_gate_activations = [keys_gate_activations[swt_id]]
                            word_values_gate_activations = [values_gate_activations[swt_id]]

                            word_keys_gate_activations_features = \
                                [hoyer_sparseness(keys_gate_activations_token_features[swt_id])]
                            word_values_gate_activations_features = \
                                [hoyer_sparseness(values_gate_activations_token_features[swt_id])]

                    if not swt.endswith('@@'):
                        open_sw = False

                        if word_keys_gate_activations is not None:
                            pos_tags.append(word_pos)
                            pos_keys_gate_activations.append(np.mean(word_keys_gate_activations))
                            pos_values_gate_activations.append(np.mean(word_values_gate_activations))

                            pos_keys_gate_activations_features.append(np.mean(word_keys_gate_activations_features))
                            pos_values_gate_activations_features.append(np.mean(word_values_gate_activations_features))

                            # Reset
                            word_keys_gate_activations = None
                            word_values_gate_activations = None

                            word_keys_gate_activations_features = None
                            word_values_gate_activations_features = None

                        else:
                            pos_tags.append(sw_pos_tags[swt_id])
                            pos_keys_gate_activations.append(keys_gate_activations[swt_id])
                            pos_values_gate_activations.append(values_gate_activations[swt_id])

                            pos_keys_gate_activations_features.append(
                                hoyer_sparseness(keys_gate_activations_token_features[swt_id]))
                            pos_values_gate_activations_features.append(
                                hoyer_sparseness(values_gate_activations_token_features[swt_id]))

                # Document and store correct predictions
                for tag_id, pos_tag in enumerate(pos_tags):
                    # Update dicts
                    if pos_tag.startswith('N'):
                        super_tag = 'NOUN'
                    elif pos_tag.startswith('V'):
                        super_tag = 'VERB'
                    elif pos_tag.startswith('A') and (eval_lang == 'de' or eval_lang == 'ru'):
                        if eval_lang == 'ru':
                            super_tag = 'ADJ/ADV'
                        else:
                            if pos_tag.startswith('ADJ') or pos_tag == 'ADV':
                                super_tag = 'ADJ/ADV'
                            else:
                                super_tag = 'OTHER'
                    elif (pos_tag.startswith('J') or pos_tag.startswith('RB')) and eval_lang == 'en':
                        super_tag = 'ADJ/ADV'
                    else:
                        super_tag = 'OTHER'

                    token_key_activation = pos_keys_gate_activations[tag_id]
                    token_value_activation = pos_values_gate_activations[tag_id]

                    # Update dicts
                    keys_pos_matches[super_tag].append(token_key_activation)
                    values_pos_matches[super_tag].append(token_value_activation)

                    token_key_sparseness = pos_keys_gate_activations_features[tag_id]
                    token_value_sparseness = pos_values_gate_activations_features[tag_id]
                    keys_pos_sparseness[super_tag].append(token_key_sparseness)
                    values_pos_sparseness[super_tag].append(token_value_sparseness)

                for tag_id, freq_tag in enumerate(freq_tags):
                    freq_tag = int(freq_tag)
                    token_key_activation = keys_gate_activations[tag_id]
                    token_value_activation = values_gate_activations[tag_id]
                    token_key_sparseness = keys_gate_activations_features[tag_id]
                    token_value_sparseness = values_gate_activations_features[tag_id]
                    if freq_tag not in keys_freq_matches.keys():
                        keys_freq_matches[freq_tag] = [token_key_activation]
                        values_freq_matches[freq_tag] = [token_value_activation]

                        keys_freq_sparseness[freq_tag] = [token_key_sparseness]
                        values_freq_sparseness[freq_tag] = [token_value_sparseness]

                    else:
                        keys_freq_matches[freq_tag].append(token_key_activation)
                        values_freq_matches[freq_tag].append(token_value_activation)

                        keys_freq_sparseness[freq_tag].append(token_key_sparseness)
                        values_freq_sparseness[freq_tag].append(token_value_sparseness)

            seen_batches += 1

            # Report
            if seen_batches % 100 == 0:
                logging.info('Evaluated {:d} batches'.format(seen_batches))

        except tf.errors.OutOfRangeError:
            break

    # Generate heat-maps for key and value activations per layer
    list_names = \
        ['keys_pos_activations', 'values_pos_activations', 'keys_frequency_activations', 'values_frequency_activations']
    for list_id, match_list in enumerate(
            [all_keys_pos_matches, all_values_pos_matches, all_keys_freq_matches, all_values_freq_matches]):
        activation_matrix = list()
        layer_ids = ['layer {:d}'.format(l_id) for l_id in range(1, config.num_encoder_layers + 1)]
        categories = None
        # Compile activation matrix
        for layer_id in range(config.num_encoder_layers):
            curr_dict = match_list[layer_id]
            categories = list(curr_dict.keys()) if categories is None else categories
            layer_activations = [np.mean(curr_dict[cat_key]) for cat_key in categories]
            activation_matrix.append(layer_activations)
        # Plot and save activation matrix
        if 'freq' in list_names[list_id]:
            categories = ['bin {:d}'.format(c) for c in categories]
        activation_matrix.reverse()
        layer_ids.reverse()
        activation_matrix = np.stack(activation_matrix, axis=0)
        activation_df = pd.DataFrame(activation_matrix,
                                     index=layer_ids,
                                     columns=categories)
        if 'freq' in list_names[list_id]:
            plt.figure(figsize=(7, 5))
        else:
            plt.figure(figsize=(10, 10))

        sns.heatmap(activation_df, cmap='coolwarm', linewidths=0.25, vmin=0.0, vmax=1.0, xticklabels=True,
                    yticklabels=True, annot=True, square=True, fmt='.3f')
        plt.yticks(rotation=0)
        plot_path = os.path.join(plot_save_dir, '{:s}.png'.format(list_names[list_id]))
        plt.savefig(plot_path, dpi='figure', pad_inches=1, bbox_inches='tight')
        plt.clf()

    # Report
    logging.info('-' * 20)
    logging.info('\nPOS-WISE GATE ACTIVATIONS')
    for layer_id in range(config.num_encoder_layers):
        logging.info('keys_layer_{:d}: '.format(layer_id + 1))
        for pos_tag in all_keys_pos_matches[layer_id].keys():
            logging.info('{:s} : {:.8f}'.format(pos_tag, np.mean(all_keys_pos_matches[layer_id][pos_tag])))
        logging.info('values_layer_{:d}: '.format(layer_id + 1))
        for pos_tag in all_values_pos_matches[layer_id].keys():
            logging.info('{:s} : {:.8f}'.format(pos_tag, np.mean(all_values_pos_matches[layer_id][pos_tag])))

    logging.info('-' * 20)
    logging.info('\nFREQ-WISE GATE ACTIVATIONS')
    for layer_id in range(config.num_encoder_layers):
        logging.info('keys_layer_{:d}: '.format(layer_id + 1))
        for freq_tag in all_keys_freq_matches[layer_id].keys():
            logging.info('{:d} : {:.8f}'.format(freq_tag, np.mean(all_keys_freq_matches[layer_id][freq_tag])))
        logging.info('values_layer_{:d}: '.format(layer_id + 1))
        for freq_tag in all_values_freq_matches[layer_id].keys():
            logging.info('{:d} : {:.8f}'.format(freq_tag, np.mean(all_values_freq_matches[layer_id][freq_tag])))

    logging.info('-' * 20)
    logging.info('\nPOS-WISE GATE ACTIVATION SPARSENESS')
    for layer_id in range(config.num_encoder_layers):
        logging.info('keys_layer_{:d}: '.format(layer_id + 1))
        for pos_tag in all_keys_pos_sparseness[layer_id].keys():
            logging.info('{:s} : {:.8f}'.format(pos_tag, np.mean(all_keys_pos_sparseness[layer_id][pos_tag])))
        logging.info('values_layer_{:d}: '.format(layer_id + 1))
        for pos_tag in all_values_pos_sparseness[layer_id].keys():
            logging.info('{:s} : {:.8f}'.format(pos_tag, np.mean(all_values_pos_sparseness[layer_id][pos_tag])))

    logging.info('-' * 20)
    logging.info('\nFREQ-WISE GATE ACTIVATION SPARSENESS')
    for layer_id in range(config.num_encoder_layers):
        logging.info('keys_layer_{:d}: '.format(layer_id + 1))
        for freq_tag in all_keys_freq_sparseness[layer_id].keys():
            logging.info('{:d} : {:.8f}'.format(freq_tag, np.mean(all_keys_freq_sparseness[layer_id][freq_tag])))
        logging.info('values_layer_{:d}: '.format(layer_id + 1))
        for freq_tag in all_values_freq_sparsness[layer_id].keys():
            logging.info('{:d} : {:.8f}'.format(freq_tag, np.mean(all_values_freq_sparsness[layer_id][freq_tag])))


def embed_sim_eval(config, sess_config):
    """ Computes the similarity of lexical embeddings and layer-wise latent representations
    for the specified NMT model. """

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

    # Set up classifier
    vocab_size = source_vocab_size if config.probe_encoder else target_vocab_size
    classifier = LexicalClassifier(config, vocab_size)

    # Save model options
    config_as_dict = OrderedDict(sorted(vars(config).items()))
    json.dump(config_as_dict, open('{:s}.json'.format(config.save_to), 'w'), indent=2)

    # Initialize session
    sess = tf.Session(config=sess_config)
    if config.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root=None)
        sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)

    # Get validation and translation OPs
    if config.num_gpus >= 2:
        validation_ops = \
            get_parallel_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training', True)
        logging.info('[Parallel training, gradient delay == {:d}]'.format(config.gradient_delay))
    else:
        validation_ops = \
            get_single_ops(model, valid_iterator, config.num_gpus, source_to_index['<EOS>'], 'training', True)
        logging.info('[Single-device training, gradient delay == {:d}]'.format(config.gradient_delay))

    # Unpack validation and translation OPs
    _, batch_loss_op, _, _ = validation_ops
    _, cls_batch_loss_op, cls_batch_accuracy_op, _ = classifier.train_model()

    logging.info('-' * 20)
    model_size = count_parameters()
    logging.info('Number of in-graph parameters (without activations): {:d}'.format(int(model_size)))
    logging.info('-' * 20)

    # Prepare model
    cls_saver, cls_checkpoint_path, saver, checkpoint_path, progress = \
        session_setup(config, sess, classifier, training=False, max_checkpoints=config.max_checkpoints)

    logging.info('-' * 20)
    if checkpoint_path is not None:
        logging.info('NMT model restored from checkpoint {:s}'.format(checkpoint_path))
    else:
        logging.info('No checkpoint to initialize the NMT model from could be found. Exiting.')
        sys.exit(1)

    logging.info('-' * 20)
    logging.info('Performing lexical similarity evaluation on corpus {:s}'
                 .format(config.valid_target_dataset, model.name))
    logging.info('[BEGIN LEXICAL SIMILARITY EVALUATION]')
    logging.info('-' * 20)

    sess.run(valid_iterator.initializer)

    # Initialize cosine similarity map
    enc_cosine_sims = {'layer_{:d}'.format(layer_id): list() for layer_id in range(config.num_encoder_layers + 1)}
    dec_cosine_sims = {'layer_{:d}'.format(layer_id): list() for layer_id in range(config.num_decoder_layers + 1)}

    stored_batches = 0
    num_sentences = 0

    model_feed_dict = {model.training: False}

    logging.info('Estimating similarity ... ')
    while True:
        # Run a forward pass through the model and collect activations
        try:
            # Get model activations
            _, enc_layer_outs, dec_layer_outs = \
                sess.run([batch_loss_op, model.enc.layer_outputs, model.dec.layer_outputs], feed_dict=model_feed_dict)

            # Unpack
            enc_lexical = enc_layer_outs['layer_0'][0]
            dec_lexical = dec_layer_outs['layer_0'][0]
            enc_contexts = \
                [enc_layer_outs['layer_{:d}'.format(layer_id + 1)][0] for layer_id in range(config.num_encoder_layers)]
            dec_contexts = \
                [dec_layer_outs['layer_{:d}'.format(layer_id + 1)][0] for layer_id in range(config.num_decoder_layers)]

            all_enc_lexical = enc_lexical
            all_dec_lexical = dec_lexical
            all_enc_contexts = enc_contexts
            all_dec_contexts = dec_contexts

            # Compare encoder
            for ctx_id in range(config.num_encoder_layers + 1):
                if ctx_id > 0:
                    curr_enc_ctx = all_enc_contexts[ctx_id - 1]
                else:
                    curr_enc_ctx = None
                sent_enc_lexical = np.reshape(all_enc_lexical[:, :-1, :], [-1])

                if ctx_id == 0:
                    sent_enc_context = sent_enc_lexical  # sanity check
                else:
                    sent_enc_context = np.reshape(curr_enc_ctx[:, :-1, :], [-1])

                # Calculate cosine similarity
                enc_sent_cosine = np.dot(sent_enc_lexical, sent_enc_context) / \
                                  (np.linalg.norm(sent_enc_lexical) * np.linalg.norm(sent_enc_context))
                enc_cosine_sims['layer_{:d}'.format(ctx_id)].append(enc_sent_cosine)

            # Compare decoder
            for ctx_id in range(config.num_decoder_layers + 1):
                if ctx_id > 0:
                    curr_dec_ctx = all_dec_contexts[ctx_id - 1]
                else:
                    curr_dec_ctx = None
                sent_dec_lexical = np.reshape(all_dec_lexical[:, 1:, :], [-1])

                if ctx_id == 0:
                    sent_dec_context = sent_dec_lexical  # sanity check
                else:
                    sent_dec_context = np.reshape(curr_dec_ctx[:, 1:, :], [-1])

                # Cosine sim
                dec_sent_cosine = np.dot(sent_dec_lexical, sent_dec_context) / \
                                  (np.linalg.norm(sent_dec_lexical) * np.linalg.norm(sent_dec_context))
                dec_cosine_sims['layer_{:d}'.format(ctx_id)].append(dec_sent_cosine)

            stored_batches += 1
            num_sentences += enc_lexical.shape[0]

            # Report
            if stored_batches % 20 == 0:
                logging.info('Evaluated {:d} batches ({:d} sentences).'
                             .format(stored_batches, num_sentences))

        except tf.errors.OutOfRangeError:
            break

    # Report
    logging.info('-' * 20)
    logging.info('Evaluation completed!\n')
    logging.info('ENCODER RESULTS:')
    logging.info('COSINE SIMILARITY:')
    for ctx_id in range(config.num_encoder_layers + 1):
        logging.info('layer {:d}: {:.8f}'
                     .format(ctx_id, np.mean(enc_cosine_sims['layer_{:d}'.format(ctx_id)])))

    logging.info('-' * 20)
    logging.info('DECODER RESULTS:')
    logging.info('COSINE SIMILARITY:')
    for ctx_id in range(config.num_decoder_layers + 1):
        logging.info('layer {:d}: {:.8f}'
                     .format(ctx_id, np.mean(dec_cosine_sims['layer_{:d}'.format(ctx_id)])))


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
    network.add_argument('--probe_encoder', action='store_true',
                         help='probe latent representations extracted from the encoder; if disabled, probe decoder')
    network.add_argument('--probe_layer', type=int, default=0, metavar='INT',
                         help='probe latent representations extracted from this layer within the specified sub-network '
                              '(default: %(default)s)')

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
    training.add_argument('--cls_reload', type=str, default=None, metavar='PATH',
                          help='load existing classifier from this path')
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
    validation.add_argument('--sim_eval', action='store_true',
                            help='evaluates the similarity between lexical embeddings and latent representations '
                                 'learned by a model')
    validation.add_argument('--classifier_eval', action='store_true',
                            help='evaluates correctness of classifier predictions based on word POS and binned '
                                 'sub-word frequency')
    validation.add_argument('--cls_pickle_dir', type=str, required=False, metavar='PATH',
                            help='directory for storing pickles containing classification results for lexical probing')
    validation.add_argument('--classifier_heatmaps', action='store_true',
                            help='generates heat-maps from the collected classification error pickles')
    validation.add_argument('--pos_reference', type=str, default=None, metavar='PATH',
                            help='path to the annotated POS reference file (default: %(default)s)')
    validation.add_argument('--freq_reference', type=str, default=None, metavar='PATH',
                            help='path to the annotated freq reference file (default: %(default)s)')

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
        if config.sim_eval:
            embed_sim_eval(config, sess_config)
        elif config.classifier_eval:
            pos_freq_match(config, sess_config)
        elif config.classifier_heatmaps:
            heatmap_from_pickle(config)
        else:
            train(config, sess_config)
