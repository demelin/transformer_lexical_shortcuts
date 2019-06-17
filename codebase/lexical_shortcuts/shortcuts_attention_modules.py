import tensorflow as tf

from layers import \
    get_shape_list, \
    matmul_nd, \
    FeedForwardLayer

from tensorflow.python.ops.init_ops import glorot_uniform_initializer


class MultiHeadAttentionShortcuts(object):
    """ Multi-head attention equipped with lexical shortcuts and feature-fusion. """

    def __init__(self,
                 reference_dims,
                 hypothesis_dims,
                 total_key_dims,
                 total_value_dims,
                 output_dims,
                 num_heads,
                 float_dtype,
                 dropout_attn,
                 training,
                 name=None):

        # Set attributes
        self.reference_dims = reference_dims
        self.hypothesis_dims = hypothesis_dims
        self.total_key_dims = total_key_dims
        self.total_value_dims = total_value_dims
        self.output_dims = output_dims
        self.num_heads = num_heads
        self.float_dtype = float_dtype
        self.dropout_attn = dropout_attn
        self.training = training
        self.name = name

        # Keep track of gate values
        self.key_gate = 0.
        self.value_gate = 0.

        # Check if the specified hyper-parameters are consistent
        if total_key_dims % num_heads != 0:
            raise ValueError('Specified total attention key dimensions {:d} must be divisible by the number of '
                             'attention heads {:d}'.format(total_key_dims, num_heads))
        if total_value_dims % num_heads != 0:
            raise ValueError('Specified total attention value dimensions {:d} must be divisible by the number of '
                             'attention heads {:d}'.format(total_value_dims, num_heads))

        # Instantiate parameters
        with tf.variable_scope(self.name):
            self.queries_projection = FeedForwardLayer(self.hypothesis_dims,
                                                       self.total_key_dims,
                                                       float_dtype,
                                                       dropout_rate=0.,
                                                       activation=None,
                                                       use_bias=False,
                                                       use_layer_norm=False,
                                                       training=self.training,
                                                       name='queries_projection')

            self.sc_keys_projection = FeedForwardLayer(self.reference_dims,
                                                       self.total_key_dims,
                                                       float_dtype,
                                                       dropout_rate=0.,
                                                       activation=None,
                                                       use_bias=False,
                                                       use_layer_norm=False,
                                                       training=self.training,
                                                       name='sc_keys_projection')

            self.sc_values_projection = FeedForwardLayer(self.reference_dims,
                                                         self.total_value_dims,
                                                         float_dtype,
                                                         dropout_rate=0.,
                                                         activation=None,
                                                         use_bias=False,
                                                         use_layer_norm=False,
                                                         training=self.training,
                                                         name='sc_values_projection')

            self.keys_projection = FeedForwardLayer(self.reference_dims,
                                                    self.total_key_dims,
                                                    float_dtype,
                                                    dropout_rate=0.,
                                                    activation=None,
                                                    use_bias=False,
                                                    use_layer_norm=False,
                                                    training=self.training,
                                                    name='keys_projection')

            self.values_projection = FeedForwardLayer(self.reference_dims,
                                                      self.total_value_dims,
                                                      float_dtype,
                                                      dropout_rate=0.,
                                                      activation=None,
                                                      use_bias=False,
                                                      use_layer_norm=False,
                                                      training=self.training,
                                                      name='values_projection')

            self.key_gate_bias = tf.get_variable(name='key_gate_bias',
                                                 shape=[self.total_key_dims],
                                                 dtype=tf.float32,
                                                 initializer=glorot_uniform_initializer())

            self.value_gate_bias = tf.get_variable(name='value_gate_bias',
                                                   shape=[self.total_value_dims],
                                                   dtype=tf.float32,
                                                   initializer=glorot_uniform_initializer())

            self.context_projection = FeedForwardLayer(self.total_value_dims,
                                                       self.output_dims,
                                                       float_dtype,
                                                       dropout_rate=0.,
                                                       activation=None,
                                                       use_bias=False,
                                                       use_layer_norm=False,
                                                       training=self.training,
                                                       name='context_projection')

    def _compute_attn_inputs(self, query_context, memory_context):
        """ Computes query, key, and value tensors used by the attention function for the calculation of the
        time-dependent context representation. """
        shortcut_inputs, hidden_inputs = memory_context
        queries = self.queries_projection.forward(query_context)
        shortcut_keys = self.sc_keys_projection.forward(shortcut_inputs)
        hidden_keys = self.keys_projection.forward(hidden_inputs)
        shortcut_values = self.sc_values_projection.forward(shortcut_inputs)
        hidden_values = self.values_projection.forward(hidden_inputs)
        return queries, shortcut_keys, hidden_keys, shortcut_values, hidden_values

    def _split_among_heads(self, inputs):
        """ Splits the attention inputs among multiple heads. """
        # Retrieve the depth of the input tensor to be split (input is 3d)
        inputs_dims = get_shape_list(inputs)
        inputs_depth = inputs_dims[-1]

        # Assert the depth is compatible with the specified number of attention heads
        if isinstance(inputs_depth, int) and isinstance(self.num_heads, int):
            assert inputs_depth % self.num_heads == 0, \
                ('Attention inputs depth {:d} is not evenly divisible by the specified number of attention heads {:d}'
                 .format(inputs_depth, self.num_heads))
        split_inputs = tf.reshape(inputs, inputs_dims[:-1] + [self.num_heads, inputs_depth // self.num_heads])
        return split_inputs

    def _merge_from_heads(self, split_inputs):
        """ Inverts the _split_among_heads operation. """
        # Transpose split_inputs to perform the merge along the last two dimensions of the split input
        split_inputs = tf.transpose(split_inputs, [0, 2, 1, 3])
        # Retrieve the depth of the tensor to be merged
        split_inputs_dims = get_shape_list(split_inputs)
        split_inputs_depth = split_inputs_dims[-1]
        # Merge the depth and num_heads dimensions of split_inputs
        merged_inputs = tf.reshape(split_inputs, split_inputs_dims[:-2] + [self.num_heads * split_inputs_depth])
        return merged_inputs

    def _dot_product_attn(self, queries, keys, values, attn_mask, scaling_on):
        """ Defines the dot-product attention function; see Vasvani et al.(2017), Eq.(1). """
        # query/ key/ value have shape = [batch_size, time_steps, num_heads, num_features]
        # Tile keys and values tensors to match the number of decoding beams
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.cond(tf.greater(num_beams, 1), lambda: tf.tile(keys, [num_beams, 1, 1, 1]), lambda: keys)
        values = tf.cond(tf.greater(num_beams, 1), lambda: tf.tile(values, [num_beams, 1, 1, 1]), lambda: values)

        # Transpose split inputs
        queries = tf.transpose(queries, [0, 2, 1, 3])
        values = tf.transpose(values, [0, 2, 1, 3])
        attn_logits = tf.matmul(queries, tf.transpose(keys, [0, 2, 3, 1]))

        # Scale attention_logits by key dimensions to prevent soft-max saturation, if specified
        if scaling_on:
            key_dims = get_shape_list(keys)[-1]
            normalizer = tf.sqrt(tf.cast(key_dims, self.float_dtype))
            attn_logits /= normalizer

        # Optionally mask out positions which should not be attended to
        # attention mask should have shape=[batch, num_heads, query_length, key_length]
        # attn_logits has shape=[batch, num_heads, query_length, key_length]
        if attn_mask is not None:
            attn_mask = tf.cond(tf.greater(num_beams, 1),
                                lambda: tf.tile(attn_mask, [num_beams, 1, 1, 1]),
                                lambda: attn_mask)
            attn_logits += attn_mask

        # Calculate attention weights
        attn_weights = tf.nn.softmax(attn_logits)

        # Optionally apply dropout:
        if self.dropout_attn > 0.0:
            attn_weights = tf.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)
        # Weigh attention values
        weighted_memories = tf.matmul(attn_weights, values)
        return weighted_memories

    def forward(self, query_context, memory_context, attn_mask, layer_memories):
        """ Propagates the input information through the attention layer. """
        # Only keep the embedding layer and the top-most encoder layer
        memory_context = [memory_context[0], memory_context[-1]]

        # Get attention inputs
        queries, shortcut_keys, hidden_keys, shortcut_values, hidden_values = \
            self._compute_attn_inputs(query_context, memory_context)

        # Apply gating (GRU-style)
        self.key_gate = tf.sigmoid(shortcut_keys + hidden_keys + self.key_gate_bias)
        self.value_gate = tf.sigmoid(shortcut_values + hidden_values + self.value_gate_bias)
        keys = self.key_gate * shortcut_keys + (1 - self.key_gate) * hidden_keys
        values = self.value_gate * shortcut_values + (1 - self.value_gate) * hidden_values

        # Recall and update memories (analogous to the RNN state) - decoder only
        if layer_memories is not None:
            keys = tf.concat([layer_memories['keys'], keys], axis=1)
            values = tf.concat([layer_memories['values'], values], axis=1)
            layer_memories['keys'] = keys
            layer_memories['values'] = values

        # Split attention inputs among attention heads
        split_queries = self._split_among_heads(queries)
        split_keys = self._split_among_heads(keys)
        split_values = self._split_among_heads(values)

        # Apply attention function
        split_weighted_memories = self._dot_product_attn(split_queries, split_keys, split_values, attn_mask,
                                                         scaling_on=True)
        # Merge head output
        weighted_memories = self._merge_from_heads(split_weighted_memories)
        # Feed through a dense layer
        projected_memories = self.context_projection.forward(weighted_memories)
        return projected_memories, layer_memories


class MultiHeadAttentionShortcutsFeatureFusion(object):
    """ Multi-head attention equipped with lexical shortcuts and feature-fusion. """

    def __init__(self,
                 reference_dims,
                 hypothesis_dims,
                 total_key_dims,
                 total_value_dims,
                 output_dims,
                 num_heads,
                 float_dtype,
                 dropout_attn,
                 training,
                 name=None):

        # Set attributes
        self.reference_dims = reference_dims
        self.hypothesis_dims = hypothesis_dims
        self.total_key_dims = total_key_dims
        self.total_value_dims = total_value_dims
        self.output_dims = output_dims
        self.num_heads = num_heads
        self.float_dtype = float_dtype
        self.dropout_attn = dropout_attn
        self.training = training
        self.name = name

        # Keep track of gate values
        self.key_gate = 0.
        self.value_gate = 0.

        # Check if the specified hyper-parameters are consistent
        if total_key_dims % num_heads != 0:
            raise ValueError('Specified total attention key dimensions {:d} must be divisible by the number of '
                             'attention heads {:d}'.format(total_key_dims, num_heads))
        if total_value_dims % num_heads != 0:
            raise ValueError('Specified total attention value dimensions {:d} must be divisible by the number of '
                             'attention heads {:d}'.format(total_value_dims, num_heads))

        # Instantiate parameters
        with tf.variable_scope(self.name):
            self.queries_projection = FeedForwardLayer(self.hypothesis_dims,
                                                       self.total_key_dims,
                                                       float_dtype,
                                                       dropout_rate=0.,
                                                       activation=None,
                                                       use_bias=False,
                                                       use_layer_norm=False,
                                                       training=self.training,
                                                       name='queries_projection')

            self.keys_projection = FeedForwardLayer(self.reference_dims * 2,
                                                    self.total_key_dims * 2,
                                                    float_dtype,
                                                    dropout_rate=0.,
                                                    activation=None,
                                                    use_bias=False,
                                                    use_layer_norm=False,
                                                    training=self.training,
                                                    name='keys_projection')

            self.values_projection = FeedForwardLayer(self.reference_dims * 2,
                                                      self.total_value_dims * 2,
                                                      float_dtype,
                                                      dropout_rate=0.,
                                                      activation=None,
                                                      use_bias=False,
                                                      use_layer_norm=False,
                                                      training=self.training,
                                                      name='values_projection')

            self.key_gate_bias = tf.get_variable(name='key_gate_bias',
                                                 shape=[self.total_key_dims],
                                                 dtype=tf.float32,
                                                 initializer=glorot_uniform_initializer())

            self.value_gate_bias = tf.get_variable(name='value_gate_bias',
                                                   shape=[self.total_value_dims],
                                                   dtype=tf.float32,
                                                   initializer=glorot_uniform_initializer())

            self.context_projection = FeedForwardLayer(self.total_value_dims,
                                                       self.output_dims,
                                                       float_dtype,
                                                       dropout_rate=0.,
                                                       activation=None,
                                                       use_bias=False,
                                                       use_layer_norm=False,
                                                       training=self.training,
                                                       name='context_projection')

    def _compute_attn_inputs(self, query_context, memory_context):
        """ Computes query, key, and value tensors used by the attention function for the calculation of the
        time-dependent context representation. """
        queries = self.queries_projection.forward(query_context)
        keys = self.keys_projection.forward(memory_context)
        values = self.values_projection.forward(memory_context)
        return queries, keys, values

    def _split_among_heads(self, inputs):
        """ Splits the attention inputs among multiple heads. """
        # Retrieve the depth of the input tensor to be split (input is 3d)
        inputs_dims = get_shape_list(inputs)
        inputs_depth = inputs_dims[-1]

        # Assert the depth is compatible with the specified number of attention heads
        if isinstance(inputs_depth, int) and isinstance(self.num_heads, int):
            assert inputs_depth % self.num_heads == 0, \
                ('Attention inputs depth {:d} is not evenly divisible by the specified number of attention heads {:d}'
                 .format(inputs_depth, self.num_heads))
        split_inputs = tf.reshape(inputs, inputs_dims[:-1] + [self.num_heads, inputs_depth // self.num_heads])
        return split_inputs

    def _merge_from_heads(self, split_inputs):
        """ Inverts the _split_among_heads operation. """
        # Transpose split_inputs to perform the merge along the last two dimensions of the split input
        split_inputs = tf.transpose(split_inputs, [0, 2, 1, 3])
        # Retrieve the depth of the tensor to be merged
        split_inputs_dims = get_shape_list(split_inputs)
        split_inputs_depth = split_inputs_dims[-1]
        # Merge the depth and num_heads dimensions of split_inputs
        merged_inputs = tf.reshape(split_inputs, split_inputs_dims[:-2] + [self.num_heads * split_inputs_depth])
        return merged_inputs

    def _dot_product_attn(self, queries, keys, values, attn_mask, scaling_on):
        """ Defines the dot-product attention function; see Vasvani et al.(2017), Eq.(1). """
        # query/ key/ value have shape = [batch_size, time_steps, num_heads, num_features]
        # Tile keys and values tensors to match the number of decoding beams
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.cond(tf.greater(num_beams, 1), lambda: tf.tile(keys, [num_beams, 1, 1, 1]), lambda: keys)
        values = tf.cond(tf.greater(num_beams, 1), lambda: tf.tile(values, [num_beams, 1, 1, 1]), lambda: values)

        # Transpose split inputs
        queries = tf.transpose(queries, [0, 2, 1, 3])
        values = tf.transpose(values, [0, 2, 1, 3])
        attn_logits = tf.matmul(queries, tf.transpose(keys, [0, 2, 3, 1]))

        # Scale attention_logits by key dimensions to prevent softmax saturation, if specified
        if scaling_on:
            key_dims = get_shape_list(keys)[-1]
            normalizer = tf.sqrt(tf.cast(key_dims, self.float_dtype))
            attn_logits /= normalizer

        # Optionally mask out positions which should not be attended to
        # attention mask should have shape=[batch, num_heads, query_length, key_length]
        # attn_logits has shape=[batch, num_heads, query_length, key_length]
        if attn_mask is not None:
            attn_mask = tf.cond(tf.greater(num_beams, 1),
                                lambda: tf.tile(attn_mask, [num_beams, 1, 1, 1]),
                                lambda: attn_mask)
            attn_logits += attn_mask

        # Calculate attention weights
        attn_weights = tf.nn.softmax(attn_logits)

        # Optionally apply dropout:
        if self.dropout_attn > 0.0:
            attn_weights = tf.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)
        # Weigh attention values
        weighted_memories = tf.matmul(attn_weights, values)
        return weighted_memories

    def forward(self, query_context, memory_context, attn_mask, layer_memories):
        """ Propagates the input information through the attention layer. """
        # Only keep the embedding layer and the top-most encoder layer and concatenate their features
        memory_context = tf.concat([memory_context[0], memory_context[-1]], axis=-1)

        # Get attention inputs
        queries, keys, values = self._compute_attn_inputs(query_context, memory_context)

        # Split in two
        shortcut_keys, hidden_keys = tf.split(keys, num_or_size_splits=2, axis=-1)
        shortcut_values, hidden_values = tf.split(values, num_or_size_splits=2, axis=-1)

        # Apply gating (GRU-style)
        self.key_gate = tf.sigmoid(shortcut_keys + hidden_keys + self.key_gate_bias)
        self.value_gate = tf.sigmoid(shortcut_values + hidden_values + self.value_gate_bias)
        keys = self.key_gate * shortcut_keys + (1 - self.key_gate) * hidden_keys
        values = self.value_gate * shortcut_values + (1 - self.value_gate) * hidden_values

        # Recall and update memories (analogous to the RNN state) - decoder only
        if layer_memories is not None:
            keys = tf.concat([layer_memories['keys'], keys], axis=1)
            values = tf.concat([layer_memories['values'], values], axis=1)
            layer_memories['keys'] = keys
            layer_memories['values'] = values

        # Split attention inputs among attention heads
        split_queries = self._split_among_heads(queries)
        split_keys = self._split_among_heads(keys)
        split_values = self._split_among_heads(values)

        # Apply attention function
        split_weighted_memories = self._dot_product_attn(split_queries, split_keys, split_values, attn_mask,
                                                         scaling_on=True)
        # Merge head output
        weighted_memories = self._merge_from_heads(split_weighted_memories)
        # Feed through a dense layer
        projected_memories = self.context_projection.forward(weighted_memories)

        return projected_memories, layer_memories


class MultiHeadAttentionShortcutsFeatureFusionNonLexical(object):
    """ Multi-head attention equipped with lexical shortcuts and feature-fusion. """

    def __init__(self,
                 reference_dims,
                 hypothesis_dims,
                 total_key_dims,
                 total_value_dims,
                 output_dims,
                 num_heads,
                 float_dtype,
                 dropout_attn,
                 training,
                 name=None):

        # Set attributes
        self.reference_dims = reference_dims
        self.hypothesis_dims = hypothesis_dims
        self.total_key_dims = total_key_dims
        self.total_value_dims = total_value_dims
        self.output_dims = output_dims
        self.num_heads = num_heads
        self.float_dtype = float_dtype
        self.dropout_attn = dropout_attn
        self.training = training
        self.name = name

        # Keep track of gate values
        self.key_gate = 0.
        self.value_gate = 0.

        # Check if the specified hyper-parameters are consistent
        if total_key_dims % num_heads != 0:
            raise ValueError('Specified total attention key dimensions {:d} must be divisible by the number of '
                             'attention heads {:d}'.format(total_key_dims, num_heads))
        if total_value_dims % num_heads != 0:
            raise ValueError('Specified total attention value dimensions {:d} must be divisible by the number of '
                             'attention heads {:d}'.format(total_value_dims, num_heads))

        # Instantiate parameters
        with tf.variable_scope(self.name):
            self.queries_projection = FeedForwardLayer(self.hypothesis_dims,
                                                       self.total_key_dims,
                                                       float_dtype,
                                                       dropout_rate=0.,
                                                       activation=None,
                                                       use_bias=False,
                                                       use_layer_norm=False,
                                                       training=self.training,
                                                       name='queries_projection')

            self.keys_projection = FeedForwardLayer(self.reference_dims * 2,
                                                    self.total_key_dims * 2,
                                                    float_dtype,
                                                    dropout_rate=0.,
                                                    activation=None,
                                                    use_bias=False,
                                                    use_layer_norm=False,
                                                    training=self.training,
                                                    name='keys_projection')

            self.values_projection = FeedForwardLayer(self.reference_dims * 2,
                                                      self.total_value_dims * 2,
                                                      float_dtype,
                                                      dropout_rate=0.,
                                                      activation=None,
                                                      use_bias=False,
                                                      use_layer_norm=False,
                                                      training=self.training,
                                                      name='values_projection')

            self.key_gate_bias = tf.get_variable(name='key_gate_bias',
                                                 shape=[self.total_key_dims],
                                                 dtype=tf.float32,
                                                 initializer=glorot_uniform_initializer())

            self.value_gate_bias = tf.get_variable(name='value_gate_bias',
                                                   shape=[self.total_value_dims],
                                                   dtype=tf.float32,
                                                   initializer=glorot_uniform_initializer())

            self.context_projection = FeedForwardLayer(self.total_value_dims,
                                                       self.output_dims,
                                                       float_dtype,
                                                       dropout_rate=0.,
                                                       activation=None,
                                                       use_bias=False,
                                                       use_layer_norm=False,
                                                       training=self.training,
                                                       name='context_projection')

    def _compute_attn_inputs(self, query_context, memory_context):
        """ Computes query, key, and value tensors used by the attention function for the calculation of the
        time-dependent context representation. """
        queries = self.queries_projection.forward(query_context)
        keys = self.keys_projection.forward(memory_context)
        values = self.values_projection.forward(memory_context)
        return queries, keys, values

    def _split_among_heads(self, inputs):
        """ Splits the attention inputs among multiple heads. """
        # Retrieve the depth of the input tensor to be split (input is 3d)
        inputs_dims = get_shape_list(inputs)
        inputs_depth = inputs_dims[-1]

        # Assert the depth is compatible with the specified number of attention heads
        if isinstance(inputs_depth, int) and isinstance(self.num_heads, int):
            assert inputs_depth % self.num_heads == 0, \
                ('Attention inputs depth {:d} is not evenly divisible by the specified number of attention heads {:d}'
                 .format(inputs_depth, self.num_heads))
        split_inputs = tf.reshape(inputs, inputs_dims[:-1] + [self.num_heads, inputs_depth // self.num_heads])
        return split_inputs

    def _merge_from_heads(self, split_inputs):
        """ Inverts the _split_among_heads operation. """
        # Transpose split_inputs to perform the merge along the last two dimensions of the split input
        split_inputs = tf.transpose(split_inputs, [0, 2, 1, 3])
        # Retrieve the depth of the tensor to be merged
        split_inputs_dims = get_shape_list(split_inputs)
        split_inputs_depth = split_inputs_dims[-1]
        # Merge the depth and num_heads dimensions of split_inputs
        merged_inputs = tf.reshape(split_inputs, split_inputs_dims[:-2] + [self.num_heads * split_inputs_depth])
        return merged_inputs

    def _dot_product_attn(self, queries, keys, values, attn_mask, scaling_on):
        """ Defines the dot-product attention function; see Vasvani et al.(2017), Eq.(1). """
        # query/ key/ value have shape = [batch_size, time_steps, num_heads, num_features]
        # Tile keys and values tensors to match the number of decoding beams
        num_beams = get_shape_list(queries)[0] // get_shape_list(keys)[0]
        keys = tf.cond(tf.greater(num_beams, 1), lambda: tf.tile(keys, [num_beams, 1, 1, 1]), lambda: keys)
        values = tf.cond(tf.greater(num_beams, 1), lambda: tf.tile(values, [num_beams, 1, 1, 1]), lambda: values)

        # Transpose split inputs
        queries = tf.transpose(queries, [0, 2, 1, 3])
        values = tf.transpose(values, [0, 2, 1, 3])
        attn_logits = tf.matmul(queries, tf.transpose(keys, [0, 2, 3, 1]))

        # Scale attention_logits by key dimensions to prevent softmax saturation, if specified
        if scaling_on:
            key_dims = get_shape_list(keys)[-1]
            normalizer = tf.sqrt(tf.cast(key_dims, self.float_dtype))
            attn_logits /= normalizer

        # Optionally mask out positions which should not be attended to
        # attention mask should have shape=[batch, num_heads, query_length, key_length]
        # attn_logits has shape=[batch, num_heads, query_length, key_length]
        if attn_mask is not None:
            attn_mask = tf.cond(tf.greater(num_beams, 1),
                                lambda: tf.tile(attn_mask, [num_beams, 1, 1, 1]),
                                lambda: attn_mask)
            attn_logits += attn_mask

        # Calculate attention weights
        attn_weights = tf.nn.softmax(attn_logits)

        # Optionally apply dropout:
        if self.dropout_attn > 0.0:
            attn_weights = tf.layers.dropout(attn_weights, rate=self.dropout_attn, training=self.training)
        # Weigh attention values
        weighted_memories = tf.matmul(attn_weights, values)
        return weighted_memories

    def forward(self, query_context, memory_context, attn_mask, layer_memories):
        """ Propagates the input information through the attention layer. """
        # Only keep the second-to-last layer and the top-most layer
        if len(memory_context) > 1:
            memory_context = tf.concat([memory_context[-2], memory_context[-1]], axis=-1)
        else:
            memory_context = tf.concat([memory_context[0], memory_context[-1]], axis=-1)

        # Get attention inputs
        queries, keys, values = self._compute_attn_inputs(query_context, memory_context)

        # Split in two
        shortcut_keys, hidden_keys = tf.split(keys, num_or_size_splits=2, axis=-1)
        shortcut_values, hidden_values = tf.split(values, num_or_size_splits=2, axis=-1)

        # Apply gating (GRU-style)
        self.key_gate = tf.sigmoid(shortcut_keys + hidden_keys + self.key_gate_bias)
        self.value_gate = tf.sigmoid(shortcut_values + hidden_values + self.value_gate_bias)
        keys = self.key_gate * shortcut_keys + (1 - self.key_gate) * hidden_keys
        values = self.value_gate * shortcut_values + (1 - self.value_gate) * hidden_values

        # Recall and update memories (analogous to the RNN state) - decoder only
        if layer_memories is not None:
            keys = tf.concat([layer_memories['keys'], keys], axis=1)
            values = tf.concat([layer_memories['values'], values], axis=1)
            layer_memories['keys'] = keys
            layer_memories['values'] = values

        # Split attention inputs among attention heads
        split_queries = self._split_among_heads(queries)
        split_keys = self._split_among_heads(keys)
        split_values = self._split_among_heads(values)

        # Apply attention function
        split_weighted_memories = self._dot_product_attn(split_queries, split_keys, split_values, attn_mask,
                                                         scaling_on=True)
        # Merge head output
        weighted_memories = self._merge_from_heads(split_weighted_memories)
        # Feed through a dense layer
        projected_memories = self.context_projection.forward(weighted_memories)

        return projected_memories, layer_memories
