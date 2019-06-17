from layers import ProcessingLayer

from lexical_shortcuts.shortcuts_attention_modules import \
    MultiHeadAttentionShortcuts, \
    MultiHeadAttentionShortcutsFeatureFusion, \
    MultiHeadAttentionShortcutsFeatureFusionNonLexical


class ShortcutsAttentionBlock(object):
    """ Defines a single attention block (referred to as 'sub-layer' in the paper) comprising of a single multi-head
    attention layer preceded by a pre-processing layer and followed by a post-processing layer. """

    def __init__(self,
                 config,
                 float_dtype,
                 self_attention,
                 training,
                 shortcut_type):

        # Set attributes
        self.config = config
        self.self_attention = self_attention
        self.shortcut_type = shortcut_type

        # Track gate values
        self.key_gate = 0.
        self.value_gate = 0.

        if self_attention:
            attn_name = 'self_attn'
        else:
            attn_name = 'cross_attn'

        memory_size = config.hidden_size

        assert shortcut_type in ['lexical', 'lexical_plus_feature_fusion', 'non-lexical'], \
            'Shortcut type {:s} is not supported.'.format(shortcut_type)

        # Build layers
        self.pre_sub_layer = ProcessingLayer(config.hidden_size,
                                             use_layer_norm=True,
                                             dropout_rate=0.0,
                                             training=training,
                                             name='pre_{:s}_sublayer'.format(attn_name))

        self.post_sub_layer = ProcessingLayer(config.hidden_size,
                                              use_layer_norm=False,
                                              dropout_rate=config.dropout_residual,
                                              training=training,
                                              name='post_{:s}_sublayer'.format(attn_name))

        if shortcut_type == 'lexical_plus_feature_fusion':
            self.attn = MultiHeadAttentionShortcutsFeatureFusion(memory_size,
                                                                 config.hidden_size,
                                                                 config.hidden_size,
                                                                 config.hidden_size,
                                                                 config.hidden_size,
                                                                 config.num_heads,
                                                                 float_dtype,
                                                                 dropout_attn=config.dropout_attn,
                                                                 training=training,
                                                                 name='{:s}_sublayer'.format(attn_name))
        elif shortcut_type == 'non_lexical':
            self.attn = MultiHeadAttentionShortcutsFeatureFusionNonLexical(memory_size,
                                                                           config.hidden_size,
                                                                           config.hidden_size,
                                                                           config.hidden_size,
                                                                           config.hidden_size,
                                                                           config.num_heads,
                                                                           float_dtype,
                                                                           dropout_attn=config.dropout_attn,
                                                                           training=training,
                                                                           name='{:s}_sublayer'.format(attn_name))
        else:
            self.attn = MultiHeadAttentionShortcuts(memory_size,
                                                    config.hidden_size,
                                                    config.hidden_size,
                                                    config.hidden_size,
                                                    config.hidden_size,
                                                    config.num_heads,
                                                    float_dtype,
                                                    dropout_attn=config.dropout_attn,
                                                    training=training,
                                                    name='{:s}_sublayer'.format(attn_name))

    def forward(self, inputs, memory_context, attn_mask, layer_memories=None):
        """ Propagates input data through the block. """
        assert (memory_context is not None), 'State cache has to be provided for the application of shortcuts.'
        # Pre-process inputs
        inputs = self.pre_sub_layer.forward(inputs)
        outputs, layer_memories = self.attn.forward(inputs, memory_context, attn_mask, layer_memories)
        # Post-process outputs
        block_out = self.post_sub_layer.forward(outputs, residual_inputs=inputs)

        # Optionally track gate values
        if self.config.track_gate_values:
            self.key_gate = self.attn.key_gate
            self.value_gate = self.attn.value_gate

        return block_out, layer_memories
