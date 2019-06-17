import tensorflow as tf

from layers import \
    EmbeddingLayer, \
    MaskedCrossEntropy, \
    get_shape_list, \
    get_right_context_mask, \
    get_positional_signal

from blocks import AttentionBlock, FFNBlock
from lexical_shortcuts.lexical_shortcuts_blocks import ShortcutsAttentionBlock
from inference import greedy_search, beam_search


class Transformer(object):
    """ The main transformer model class. """

    def __init__(self, config, source_vocab_size, target_vocab_size, name):
        # Set attributes
        self.config = config
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.name = name
        self.int_dtype = tf.int32
        self.float_dtype = tf.float32

        with tf.name_scope('{:s}_inputs_and_variables'.format(self.name)), tf.device('/cpu:0'):
            # Declare placeholders
            self.learning_rate = tf.placeholder(dtype=self.float_dtype, name='adaptive_learning_rate', shape=[])
            self.training = tf.placeholder_with_default(False, name='is_training', shape=[])
            # Instantiate global step variable (used for training)
            self.global_step = tf.get_variable(name='global_step',
                                               shape=[],
                                               dtype=tf.int32,
                                               initializer=tf.zeros_initializer,
                                               trainable=False)
            # Input attributes
            self.source_ids = None
            self.target_ids_in = None
            self.target_ids_out = None
            self.source_mask = None
            self.target_mask = None

            self.optimizer = self._get_optimizer()

            # Optionally track gate values
            self.gate_tracker = dict()
            if config.track_gate_values:
                dec_layers = \
                    ['decoder_layer_{:d}'.format(layer_id) for layer_id in range(1, config.num_decoder_layers + 1)]
                for layer_name in dec_layers:
                    self.gate_tracker[layer_name] = dict()
                    self.gate_tracker[layer_name]['lexical_gate_keys'] = list()
                    self.gate_tracker[layer_name]['lexical_gate_values'] = list()

    def load_global_step(self, global_step_value, session):
        """ Assigns value to the global_step variable when loading model parameters from a previous checkpoint. """
        self.global_step.load(global_step_value, session)

    def _build_graph(self):
        """ Defines the model graph. """
        with tf.variable_scope('{:s}_model'.format(self.name)):
            # Instantiate embedding layer(s)
            if self.config.untie_enc_dec_embeddings:
                enc_vocab_size = self.source_vocab_size
                dec_vocab_size = self.target_vocab_size
            else:
                assert self.source_vocab_size == self.target_vocab_size, \
                    'Input and output vocabularies should be identical when tying embedding tables.'
                enc_vocab_size = dec_vocab_size = self.source_vocab_size

            encoder_embedding_layer = EmbeddingLayer(enc_vocab_size,
                                                     self.config.embedding_size,
                                                     self.config.hidden_size,
                                                     self.float_dtype,
                                                     name='encoder_embedding_layer')
            if self.config.untie_enc_dec_embeddings:
                decoder_embedding_layer = EmbeddingLayer(dec_vocab_size,
                                                         self.config.embedding_size,
                                                         self.config.hidden_size,
                                                         self.float_dtype,
                                                         name='decoder_embedding_layer')
            else:
                decoder_embedding_layer = encoder_embedding_layer

            if self.config.untie_decoder_embeddings:
                softmax_projection_layer = EmbeddingLayer(dec_vocab_size,
                                                          self.config.embedding_size,
                                                          self.config.hidden_size,
                                                          self.float_dtype,
                                                          name='softmax_projection_layer')
            else:
                softmax_projection_layer = decoder_embedding_layer

            # Instantiate the component networks
            self.enc = TransformerEncoder(self.config,
                                          encoder_embedding_layer,
                                          self.training,
                                          self.float_dtype,
                                          'encoder')
            self.dec = TransformerDecoder(self.config,
                                          decoder_embedding_layer,
                                          softmax_projection_layer,
                                          self.training,
                                          self.int_dtype,
                                          self.float_dtype,
                                          self.gate_tracker,
                                          'decoder')

        return dec_vocab_size

    def _decode(self):
        """ Generates prediction logits at training time. """
        # (Re-)generate the computational graph
        dec_vocab_size = self._build_graph()
        # Encode source sequences
        with tf.name_scope('{:s}_encode'.format(self.name)):
            enc_output, cross_attn_mask = self.enc.encode(self.source_ids, self.source_mask)
        # Decode into target sequences
        with tf.name_scope('{:s}_decode'.format(self.name)):
            logits = self.dec.decode_at_train(self.target_ids_in, enc_output, cross_attn_mask)
        return logits, dec_vocab_size

    def _get_losses(self):
        """ Returns the training loss incurred by the model to be used in the optimization step. """
        with tf.name_scope('{:s}_loss'.format(self.name)):
            # Get logits
            logits, dec_vocab_size = self._decode()

            # Instantiate loss layer(s)
            loss_layer = MaskedCrossEntropy(dec_vocab_size,
                                            self.config.label_smoothing_discount,
                                            self.int_dtype,
                                            self.float_dtype,
                                            time_major=False,
                                            name='loss_layer')

            # Calculate loss
            masked_loss, sentence_loss, batch_loss = \
                loss_layer.forward(logits, self.target_ids_out, self.target_mask, self.training)
        return masked_loss, sentence_loss, batch_loss

    def _get_optimizer(self):
        """ Sets up the model's optimizer. """
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                      beta1=self.config.adam_beta1,
                                      beta2=self.config.adam_beta2,
                                      epsilon=self.config.adam_epsilon)

    def _optimize(self, optimization_objective):
        """ Optimizes model parameters via stochastic gradient descent. """
        with tf.name_scope('{:s}_optimization'.format(self.name)):
            # Get trainable variables
            t_vars = tf.trainable_variables()
            # Get gradients
            grads_and_vars = self.optimizer.compute_gradients(optimization_objective)
            grads, _ = zip(*grads_and_vars)
            # Optionally clip gradients to prevent saturation effects
            if self.config.grad_norm_threshold > 0.0:
                grads, _ = tf.clip_by_global_norm(grads, self.config.grad_norm_threshold)
            # Update model parameters by optimizing on obtained gradients
            train_op = self.optimizer.apply_gradients(zip(grads, t_vars), global_step=self.global_step)
        return grads_and_vars, train_op

    def get_summaries(self, batch_loss):
        """ Specifies which information is to be summarized for TensorBoard inspection. """
        with tf.name_scope('model_summaries'):
            learning_rate_summary = tf.summary.scalar(name='learning_rate', tensor=self.learning_rate)
            batch_loss_summary = tf.summary.scalar(name='mean_batch_loss', tensor=batch_loss)
            model_summaries = tf.summary.merge([learning_rate_summary, batch_loss_summary], name='base_summaries')

        if self.config.track_gate_values:
            with tf.name_scope('gate_summaries'):
                gate_summaries = list()
                for layer_name in self.gate_tracker.keys():
                    avg_lexical_gate_keys = tf.reduce_mean(self.gate_tracker[layer_name]['lexical_gate_keys'])
                    avg_lexical_gate_values = tf.reduce_mean(self.gate_tracker[layer_name]['lexical_gate_values'])
                    self.gate_tracker[layer_name]['lexical_gate_keys'] = list()
                    self.gate_tracker[layer_name]['lexical_gate_values'] = list()
                    gate_summaries.append(tf.summary.scalar(name='{:s}_lexical_gate_keys'.format(layer_name),
                                                            tensor=avg_lexical_gate_keys))
                    gate_summaries.append(tf.summary.scalar(name='{:s}_lexical_gate_values'.format(layer_name),
                                                            tensor=avg_lexical_gate_values))

                gate_summaries = tf.summary.merge(gate_summaries, name='gate_summaries')

            if len(self.gate_tracker.keys()) > 0:
                # Merge
                with tf.name_scope('model_summaries'):
                    model_summaries = tf.summary.merge([model_summaries, gate_summaries], name='model_summaries')

        return model_summaries

    def train_model(self, input_batch):
        """ Defines the model graph for the training step. """
        with tf.name_scope('{:s}_training'.format(self.name)):
            # Unpack inputs
            self.source_ids, self.target_ids_in, self.target_ids_out, self.source_mask, self.target_mask = input_batch
            # Optimize
            _, sentence_loss, batch_loss = self._get_losses()
            grads_and_vars, train_op = self._optimize(batch_loss)
            # Calculate the number of words processed at the current step
            words_processed = tf.reduce_sum(tf.ones_like(self.source_ids))
            words_evaluated = tf.reduce_sum(tf.ones_like(self.target_mask))
        return grads_and_vars, train_op, batch_loss, sentence_loss, words_processed, words_evaluated

    def decode_greedy(self, input_batch, do_sample=False, beam_size=0):
        """ Generates translation hypotheses via greedy decoding. """
        # Unpack inputs
        self.source_ids, self.target_ids_in, self.target_ids_out, self.source_mask, self.target_mask = input_batch
        # (Re-)generate the computational graph
        dec_vocab_size = self._build_graph()
        # Determine size of current batch
        batch_size, _ = get_shape_list(self.source_ids)
        # Encode source sequences
        with tf.name_scope('{:s}_encode'.format(self.name)):
            enc_output, cross_attn_mask = self.enc.encode(self.source_ids, self.source_mask)
        # Decode into target sequences
        with tf.name_scope('{:s}_decode'.format(self.name)):
            dec_output, scores, = self.dec.decode_at_test(enc_output, cross_attn_mask, batch_size, beam_size, do_sample)
        return dec_output, scores, dec_vocab_size

    def decode_with_sampling(self, input_batch):
        """ Generates translation hypotheses via weighted sampling. """
        decoded_ids, decoded_scores, _ = self.decode_greedy(input_batch, do_sample=True)
        return decoded_ids, decoded_scores

    def decode_with_beam_search(self, input_batch):
        """ Generates translation hypotheses via beam-search assisted decoding. """
        decoded_ids, decoded_scores, _ = self.decode_greedy(input_batch, beam_size=self.config.beam_size)
        return decoded_ids, decoded_scores


class TransformerEncoder(object):
    """ The encoder module used within the transformer model. """

    def __init__(self,
                 config,
                 embedding_layer,
                 training,
                 float_dtype,
                 name):
        # Set attributes
        self.config = config
        self.embedding_layer = embedding_layer
        self.training = training
        self.float_dtype = float_dtype
        self.name = name

        # Track layers
        self.encoder_stack = dict()
        self.is_final_layer = False

    def _embed(self, index_sequence):
        """ Embeds source-side indices to obtain the corresponding dense tensor representations. """
        # Embed input tokens
        return self.embedding_layer.embed(index_sequence)

    def _build_graph(self):
        """ Defines the model graph. """
        # Initialize layers
        with tf.variable_scope(self.name):
            for layer_id in range(1, self.config.num_encoder_layers + 1):
                layer_name = 'layer_{:d}'.format(layer_id)
                # Check if constructed layer is final
                if layer_id == self.config.num_encoder_layers:
                    self.is_final_layer = True
                # Specify ffn dimensions sequence
                ffn_dims = [self.config.ffn_hidden_size, self.config.hidden_size]
                with tf.variable_scope(layer_name):
                    # Build layer blocks (see layers.py)
                    self_attn_block = AttentionBlock(self.config,
                                                     self.float_dtype,
                                                     self_attention=True,
                                                     training=self.training)
                    ffn_block = FFNBlock(self.config,
                                         ffn_dims,
                                         self.float_dtype,
                                         is_final=self.is_final_layer,
                                         training=self.training)

                # Maintain layer-wise dict entries for easier data-passing (may change later)
                self.encoder_stack[layer_id] = dict()
                self.encoder_stack[layer_id]['self_attn'] = self_attn_block
                self.encoder_stack[layer_id]['ffn'] = ffn_block

    def encode(self, source_ids, source_mask):
        """ Encodes source-side input tokens into meaningful, contextually-enriched representations. """

        def _prepare_source():
            """ Pre-processes inputs to the encoder and generates the corresponding attention masks."""
            # Embed
            source_embeddings = self._embed(source_ids)
            # Obtain length and depth of the input tensors
            _, time_steps, depth = get_shape_list(source_embeddings)
            # Transform input mask into attention mask
            inverse_mask = tf.cast(tf.equal(source_mask, 0.0), dtype=self.float_dtype)
            attn_mask = inverse_mask * -1e9
            # Expansion to shape [batch_size, 1, 1, time_steps] is needed for compatibility with attention logits
            attn_mask = tf.expand_dims(tf.expand_dims(attn_mask, 1), 1)
            # Differentiate between self-attention and cross-attention masks for further, optional modifications
            self_attn_mask = attn_mask
            cross_attn_mask = attn_mask
            # Add positional encodings
            positional_signal = get_positional_signal(time_steps, depth, self.float_dtype)
            source_embeddings += positional_signal
            # Apply dropout
            if self.config.dropout_embeddings > 0:
                source_embeddings = tf.layers.dropout(source_embeddings,
                                                      rate=self.config.dropout_embeddings, training=self.training)
            return source_embeddings, self_attn_mask, cross_attn_mask

        with tf.variable_scope(self.name):
            # Create nodes
            self._build_graph()
            # Prepare inputs to the encoder, get attention masks
            enc_inputs, self_attn_mask, cross_attn_mask = _prepare_source()
            # Propagate inputs through the encoder stack
            enc_output = enc_inputs
            for layer_id in range(1, self.config.num_encoder_layers + 1):
                enc_output, _ = self.encoder_stack[layer_id]['self_attn'].forward(enc_output, None, self_attn_mask)
                enc_output = self.encoder_stack[layer_id]['ffn'].forward(enc_output)
        return enc_output, cross_attn_mask


class TransformerDecoder(object):
    """ The decoder module used within the transformer model. """

    def __init__(self,
                 config,
                 embedding_layer,
                 softmax_projection_layer,
                 training,
                 int_dtype,
                 float_dtype,
                 gate_tracker,
                 name):

        # Set attributes
        self.config = config
        self.embedding_layer = embedding_layer
        self.softmax_projection_layer = softmax_projection_layer
        self.training = training
        self.int_dtype = int_dtype
        self.float_dtype = float_dtype
        self.gate_tracker = gate_tracker
        self.name = name

        # If the decoder is used in a hybrid system, adjust parameters accordingly
        self.time_dim = 1

        # Track layers
        self.decoder_stack = dict()
        self.is_final_layer = False

    def _embed(self, index_sequence):
        """ Embeds target-side indices to obtain the corresponding dense tensor representations. """
        return self.embedding_layer.embed(index_sequence)

    def _get_initial_memories(self, batch_size, beam_size):
        """ Initializes decoder memories used for accelerated inference. """
        initial_memories = dict()
        for layer_id in range(1, self.config.num_decoder_layers + 1):
            initial_memories['layer_{:d}'.format(layer_id)] = \
                {'keys': tf.tile(tf.zeros([batch_size, 0, self.config.hidden_size]), [beam_size, 1, 1]),
                 'values': tf.tile(tf.zeros([batch_size, 0, self.config.hidden_size]), [beam_size, 1, 1])}
        return initial_memories

    def _build_graph(self):
        """ Defines the model graph. """
        # Initialize layers
        with tf.variable_scope(self.name):
            for layer_id in range(1, self.config.num_encoder_layers + 1):
                layer_name = 'layer_{:d}'.format(layer_id)
                # Check if constructed layer is final
                if layer_id == self.config.num_encoder_layers:
                    self.is_final_layer = True
                # Specify ffn dimensions sequence
                ffn_dims = [self.config.ffn_hidden_size, self.config.hidden_size]
                with tf.variable_scope(layer_name):
                    # Build layer blocks (see layers.py)
                    self_attn_block = ShortcutsAttentionBlock(self.config,
                                                              self.float_dtype,
                                                              self_attention=True,
                                                              training=self.training,
                                                              shortcut_type=self.config.shortcut_type)
                    cross_attn_block = AttentionBlock(self.config,
                                                      self.float_dtype,
                                                      self_attention=False,
                                                      training=self.training)
                    ffn_block = FFNBlock(self.config,
                                         ffn_dims,
                                         self.float_dtype,
                                         is_final=self.is_final_layer,
                                         training=self.training)

                # Maintain layer-wise dict entries for easier data-passing (may change later)
                self.decoder_stack[layer_id] = dict()
                self.decoder_stack[layer_id]['self_attn'] = self_attn_block
                self.decoder_stack[layer_id]['cross_attn'] = cross_attn_block
                self.decoder_stack[layer_id]['ffn'] = ffn_block

    def decode_at_train(self, target_ids, enc_output, cross_attn_mask):
        """ Returns the probability distribution over target-side tokens conditioned on the output of the encoder;
         performs decoding in parallel at training time. """

        def _decode_all(target_embeddings):
            """ Decodes the encoder-generated representations into target-side logits in parallel. """
            # Apply input dropout
            dec_input = \
                tf.layers.dropout(target_embeddings, rate=self.config.dropout_embeddings, training=self.training)
            # Maintain state cache
            state_cache = [dec_input]
            # Propagate inputs through the encoder stack
            dec_output = dec_input
            for layer_id in range(1, self.config.num_decoder_layers + 1):
                dec_output, _ = \
                    self.decoder_stack[layer_id]['self_attn'].forward(dec_output, state_cache, self_attn_mask)
                dec_output, _ = \
                    self.decoder_stack[layer_id]['cross_attn'].forward(dec_output, enc_output, cross_attn_mask)
                dec_output = self.decoder_stack[layer_id]['ffn'].forward(dec_output)
                # Cache layer-wise encoder representations
                state_cache.append(dec_output)

                # Update gate-tracker
                if len(self.gate_tracker.keys()) > 0:
                    self.gate_tracker['decoder_layer_{:d}'.format(layer_id)]['lexical_gate_keys'] = \
                        self.decoder_stack[layer_id]['self_attn'].key_gate
                    self.gate_tracker['decoder_layer_{:d}'.format(layer_id)]['lexical_gate_values'] = \
                        self.decoder_stack[layer_id]['self_attn'].value_gate

            return dec_output

        def _prepare_targets():
            """ Pre-processes target token ids before they're passed on as input to the decoder
            for parallel decoding. """
            # Embed target_ids
            target_embeddings = self._embed(target_ids)
            target_embeddings += positional_signal
            if self.config.dropout_embeddings > 0:
                target_embeddings = tf.layers.dropout(target_embeddings,
                                                      rate=self.config.dropout_embeddings, training=self.training)
            return target_embeddings

        def _decoding_function():
            """ Generates logits for target-side tokens. """
            # Embed the model's predictions up to the current time-step; add positional information, mask
            target_embeddings = _prepare_targets()
            # Pass encoder context and decoder embeddings through the decoder
            dec_output = _decode_all(target_embeddings)
            # Project decoder stack outputs and apply the soft-max non-linearity
            full_logits = self.softmax_projection_layer.project(dec_output)
            return full_logits

        with tf.variable_scope(self.name):
            # Create nodes
            self._build_graph()

            self_attn_mask = get_right_context_mask(tf.shape(target_ids)[-1])
            positional_signal = get_positional_signal(tf.shape(target_ids)[-1],
                                                      self.config.embedding_size,
                                                      self.float_dtype)
            logits = _decoding_function()
        return logits

    def decode_at_test(self, enc_output, cross_attn_mask, batch_size, beam_size, do_sample):
        """ Returns the probability distribution over target-side tokens conditioned on the output of the encoder;
         performs decoding via auto-regression at test time. """

        def _decode_step(target_embeddings, memories):
            """ Decode the encoder-generated representations into target-side logits with auto-regression. """
            # Maintain state cache
            state_cache = [target_embeddings]
            # Propagate inputs through the encoder stack
            dec_output = target_embeddings
            # NOTE: No self-attention mask is applied at decoding, as future information is unavailable
            for layer_id in range(1, self.config.num_decoder_layers + 1):
                dec_output, memories['layer_{:d}'.format(layer_id)] = \
                    self.decoder_stack[layer_id]['self_attn'].forward(
                        dec_output, state_cache, None, memories['layer_{:d}'.format(layer_id)])
                dec_output, _ = \
                    self.decoder_stack[layer_id]['cross_attn'].forward(dec_output, enc_output, cross_attn_mask)
                dec_output = self.decoder_stack[layer_id]['ffn'].forward(dec_output)
                # Cache layer-wise encoder representations
                state_cache.append(dec_output)
            # Return prediction at the final time-step to be consistent with the inference pipeline
            dec_output = dec_output[:, -1, :]
            return dec_output, memories

        def _pre_process_targets(step_target_ids, current_time_step):
            """ Pre-processes target token ids before they're passed on as input to the decoder
            for auto-regressive decoding. """
            # Embed target_ids
            target_embeddings = self._embed(step_target_ids)
            signal_slice = positional_signal[:, current_time_step - 1: current_time_step, :]
            target_embeddings += signal_slice
            if self.config.dropout_embeddings > 0:
                target_embeddings = tf.layers.dropout(target_embeddings,
                                                      rate=self.config.dropout_embeddings, training=self.training)
            return target_embeddings

        def _decoding_function(step_target_ids, current_time_step, memories):
            """ Generates logits for the target-side token predicted for the next-time step with auto-regression. """
            # Embed the model's predictions up to the current time-step; add positional information, mask
            target_embeddings = _pre_process_targets(step_target_ids, current_time_step)
            # Pass encoder context and decoder embeddings through the decoder
            dec_output, memories = _decode_step(target_embeddings, memories)
            # Project decoder stack outputs and apply the soft-max non-linearity
            step_logits = self.softmax_projection_layer.project(dec_output)
            return step_logits, memories

        with tf.variable_scope(self.name):
            # Create nodes
            self._build_graph()

            positional_signal = get_positional_signal(self.config.translation_max_len,
                                                      self.config.embedding_size,
                                                      self.float_dtype)
            if beam_size > 0:
                # Initialize target IDs with <GO>
                initial_ids = tf.cast(tf.fill([batch_size], 1), dtype=self.int_dtype)
                initial_memories = self._get_initial_memories(batch_size, beam_size=beam_size)
                output_sequences, scores = beam_search(_decoding_function,
                                                       initial_ids,
                                                       initial_memories,
                                                       self.int_dtype,
                                                       self.float_dtype,
                                                       self.config.translation_max_len,
                                                       batch_size,
                                                       beam_size,
                                                       self.embedding_layer.get_vocab_size(),
                                                       0,
                                                       self.config.length_normalization_alpha)

            else:
                # Initialize target IDs with <GO>
                initial_ids = tf.cast(tf.fill([batch_size, 1], 1), dtype=self.int_dtype)
                initial_memories = self._get_initial_memories(batch_size, beam_size=1)
                output_sequences, scores = greedy_search(_decoding_function,
                                                         initial_ids,
                                                         initial_memories,
                                                         self.int_dtype,
                                                         self.float_dtype,
                                                         self.config.translation_max_len,
                                                         batch_size,
                                                         0,
                                                         do_sample,
                                                         time_major=False)
        return output_sequences, scores
