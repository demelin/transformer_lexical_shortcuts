import tensorflow as tf

from layers import get_shape_list

""" Some inference mechanisms are adopted from the tensor2tensor library, with moderate modifications. """


# ============================================== Helper functions ==================================================== #
def batch_to_beam(batch_tensor, beam_size):
    """ Multiplies the batch tensor so as to match the size of the model's beam. """
    batch_clones = [batch_tensor] * beam_size
    new_beam = tf.stack(batch_clones, axis=1)
    return new_beam


def compute_batch_indices(batch_size, beam_size):
    """ Generates a matrix of batch indices for the 'merged' beam tensor; each index denotes the batch from which the
    sequence occupying the same relative position as the index within the 'merged' tensor belongs. """
    batch_range = tf.range(batch_size * beam_size) // beam_size
    batch_index_matrix = tf.reshape(batch_range, [batch_size, beam_size])
    return batch_index_matrix


def get_memory_invariants(memories):
    """ Calculates the invariant shapes for the model memories (i.e. layer-wise attentions of the transformer). """
    memory_type = type(memories)
    if memory_type == dict:
        memory_invariants = dict()
        for layer_id in memories.keys():
            memory_invariants[layer_id] = {key: tf.TensorShape([None] * len(get_shape_list(memories[layer_id][key])))
                                           for key in memories[layer_id].keys()}
    else:
        raise ValueError('Memory type not supported, must be a dictionary.')
    return memory_invariants


def gather_memories(memory_dict, gather_coordinates):
    """ Gathers layer-wise memory tensors corresponding to top sequences from the provided memory dictionary
    during beam search. """
    # Initialize dicts
    gathered_memories = dict()
    # Get coordinate shapes
    coords_dims = get_shape_list(gather_coordinates)

    # Gather
    for layer_key in memory_dict.keys():
        layer_dict = memory_dict[layer_key]
        gathered_memories[layer_key] = dict()

        for attn_key in layer_dict.keys():
            attn_tensor = layer_dict[attn_key]
            attn_dims = get_shape_list(attn_tensor)
            # Not sure if this is faster than the 'memory-less' version
            flat_tensor = \
                tf.transpose(tf.reshape(attn_tensor, [-1, coords_dims[0]] + attn_dims[1:]), [1, 0, 2, 3])
            gathered_values = tf.reshape(tf.transpose(tf.gather_nd(flat_tensor, gather_coordinates), [1, 0, 2, 3]),
                                         [tf.multiply(coords_dims[1], coords_dims[0])] + attn_dims[1:])
            gathered_memories[layer_key][attn_key] = gathered_values

    return gathered_memories


def gather_top_sequences(all_sequences,
                         all_scores,
                         all_scores_to_gather,
                         all_eos_flags,
                         all_memories,
                         beam_size,
                         batch_size,
                         prefix):
    """ Selects |beam size| sequences from a |beam size ** 2| sequence set; the selected sequences are used to update
    sets of unfinished and completed decoding hypotheses. """
    # Obtain indices of the top-k scores within the scores tensor
    _, top_indices = tf.nn.top_k(all_scores, k=beam_size)
    # Create a lookup-indices tensor for gathering the sequences associated with the top scores
    batch_index_matrix = compute_batch_indices(batch_size, beam_size)
    gather_coordinates = tf.stack([batch_index_matrix, top_indices], axis=2)  # coordinates in final dimension
    # Collect top outputs
    gathered_sequences = tf.gather_nd(all_sequences, gather_coordinates, name='{:s}_sequences'.format(prefix))
    gathered_scores = tf.gather_nd(all_scores_to_gather, gather_coordinates, name='{:s}_scores'.format(prefix))
    gathered_eos_flags = tf.gather_nd(all_eos_flags, gather_coordinates, name='{:s}_eos_flags'.format(prefix))
    gathered_memories = None
    if all_memories is not None:
        gathered_memories = gather_memories(all_memories, gather_coordinates)

    return gathered_sequences, gathered_scores, gathered_eos_flags, gathered_memories


# ============================================= Decoding functions =================================================== #
def greedy_search(decoding_function,
                  initial_ids,
                  initial_memories,
                  int_dtype,
                  float_dtype,
                  max_prediction_length,
                  batch_size,
                  eos_id,
                  do_sample,
                  time_major):
    """ Greedily decodes the target sequence conditioned on the output of the encoder and the current output prefix. """

    # Declare time-dimension
    time_dim = int(not time_major)  # i.e. 0 if time_major, 1 if batch_major

    # Define the 'body for the tf.while_loop() call
    def _decoding_step(current_time_step, all_finished, next_ids, decoded_ids, decoded_score, memories):
        """ Defines a single step of greedy decoding. """
        # Propagate through decoder
        step_logits, memories = decoding_function(next_ids, current_time_step, memories)
        # Calculate log probabilities for token prediction at current time-step
        step_scores = tf.nn.log_softmax(step_logits)
        # Determine next token to be generated, next_ids has shape [batch_size]
        if do_sample:
            next_ids = tf.squeeze(tf.multinomial(step_scores, num_samples=1, output_dtype=int_dtype), axis=1)
        else:
            # Greedy decoding
            next_ids = tf.argmax(step_scores, -1, output_type=int_dtype)
        # Collect scores associated with the selected tokens
        score_coordinates = tf.stack([tf.range(batch_size, dtype=int_dtype), next_ids], axis=1)
        decoded_score += tf.gather_nd(step_scores, score_coordinates)
        # Concatenate newly decoded token ID with the previously decoded ones
        decoded_ids = tf.concat([decoded_ids, tf.expand_dims(next_ids, 1)], 1)
        # Extend next_id's dimensions to be compatible with input dimensionality for the subsequent step
        next_ids = tf.expand_dims(next_ids, time_dim)
        # Check if generation has concluded with <EOS>
        # all_finished |= tf.equal(tf.squeeze(next_ids, axis=time_dim), eos_id)
        all_finished |= tf.equal(tf.reduce_prod(decoded_ids - eos_id, axis=time_dim), eos_id)

        return current_time_step + 1, all_finished, next_ids, decoded_ids, decoded_score, memories

    # Define the termination condition for the tf.while_loop() call
    def _continue_decoding(_current_time_step, _all_finished, *_):
        """ Returns 'False' if all of the sequences in the generated sequence batch exceeded the maximum specified
        length or terminated with <EOS>, upon which the while loop is exited. """
        continuation_check = \
            tf.logical_and(tf.less(_current_time_step, max_prediction_length),
                           tf.logical_not(tf.reduce_all(_all_finished)))

        return continuation_check

    # Initialize decoding-relevant variables and containers
    current_time_step = tf.constant(1)
    all_finished = tf.fill([batch_size], False)  # None of the sequences is marked as finished
    next_ids = initial_ids
    decoded_ids = tf.zeros([batch_size, 0], dtype=int_dtype)  # Sequence buffer is empty
    decoded_score = tf.zeros([batch_size], dtype=float_dtype)
    memories = initial_memories

    # Execute the auto-regressive decoding step via while loop
    _, _, _, decoded_ids, log_scores, memories = \
        tf.while_loop(cond=_continue_decoding,
                      body=_decoding_step,
                      loop_vars=[current_time_step, all_finished, next_ids, decoded_ids, decoded_score, memories],
                      shape_invariants=[tf.TensorShape([]),
                                        tf.TensorShape([None]),
                                        tf.TensorShape([None, None]),
                                        tf.TensorShape([None, None]),
                                        tf.TensorShape([None]),
                                        get_memory_invariants(memories)],
                      parallel_iterations=10,
                      swap_memory=False,
                      back_prop=False)

    # Should return logits also, for training
    return decoded_ids, log_scores


def beam_search(decoding_function,
                initial_ids,
                initial_memories,
                int_dtype,
                float_dtype,
                translation_maxlen,
                batch_size,
                beam_size,
                vocab_size,
                eos_id,
                normalization_alpha):
    """ Decodes the target sequence by maintaining a beam of candidate hypotheses, thus allowing for better exploration
    of the hypothesis space; optionally applies scaled length normalization; based on the T2T implementation.

        alive = set of n unfinished hypotheses presently within the beam; n == beam_size
        finished = set of n finished hypotheses, each terminating in <EOS>; n == beam_size

    """

    def _extend_hypotheses(current_time_step, alive_sequences, alive_log_probs, alive_memories):
        """ Generates top-k extensions of the alive beam candidates from the previous time-step, which are subsequently
        used to update the alive and finished sets at the current time-step; top-k = 2 s* beam_size """
        # Get logits for the current prediction step
        next_ids = alive_sequences[:, :, -1]  # [batch_size, beam_size]
        next_ids = tf.transpose(next_ids, [1, 0])  # [beam_size, batch_size]; transpose to match model
        next_ids = tf.reshape(next_ids, [-1, 1])  # [beam_size * batch_size, 1]

        step_logits, alive_memories = decoding_function(next_ids, current_time_step, alive_memories)
        step_logits = tf.reshape(step_logits, [beam_size, batch_size, -1])  # [beam_size, batch_size, num_words]
        step_logits = tf.transpose(step_logits, [1, 0, 2])  # [batch_size, beam_size, num_words]; transpose back

        # Calculate the scores for all possible extensions of alive hypotheses
        candidate_log_probs = tf.nn.log_softmax(step_logits, axis=-1)
        curr_log_probs = candidate_log_probs + tf.expand_dims(alive_log_probs, axis=2)

        # Apply length normalization
        length_penalty = 1.
        if normalization_alpha > 0.:
            length_penalty = ((5. + tf.to_float(current_time_step)) ** normalization_alpha) / \
                             ((5. + 1.) ** normalization_alpha)
        curr_scores = curr_log_probs / length_penalty

        # Select top-k highest scores
        flat_curr_scores = tf.reshape(curr_scores, [batch_size, -1])
        top_scores, top_ids = tf.nn.top_k(flat_curr_scores, k=beam_size ** 2)

        # Recover non-normalized scores for tracking
        top_log_probs = top_scores * length_penalty

        # Determine the beam from which the top-scoring items originate and their identity (i.e. token-ID)
        top_beam_indices = top_ids // vocab_size
        top_ids %= vocab_size

        # Determine the location of top candidates
        batch_index_matrix = compute_batch_indices(batch_size, beam_size ** 2)  # [batch_size, beam_size * factor]
        top_coordinates = tf.stack([batch_index_matrix, top_beam_indices], axis=2)

        # Extract top decoded sequences
        top_sequences = tf.gather_nd(alive_sequences, top_coordinates)  # [batch_size, beam_size * factor, sent_len]
        top_sequences = tf.concat([top_sequences, tf.expand_dims(top_ids, axis=2)], axis=2)

        # Extract top memories
        top_memories = gather_memories(alive_memories, top_coordinates)
        # top_memories = alive_memories

        # Check how many of the top sequences have terminated
        top_eos_flags = tf.equal(top_ids, eos_id)  # [batch_size, beam_size * factor]

        # Diversify beams at the outset of the generation
        init_top_sequences = tf.reshape(
            tf.reshape(top_sequences, [batch_size, beam_size, beam_size, -1])[:, :, 1, :], [batch_size, beam_size, -1])
        init_top_log_probs = \
            tf.reshape(tf.reshape(top_log_probs, [batch_size, beam_size, beam_size])[:, :, 1], [batch_size, beam_size])
        init_top_scores = \
            tf.reshape(tf.reshape(top_scores, [batch_size, beam_size, beam_size])[:, :, 1], [batch_size, beam_size])
        init_top_eos_flags = \
            tf.reshape(tf.reshape(top_eos_flags, [batch_size, beam_size, beam_size])[:, :, 1], [batch_size, beam_size])

        top_sequences, top_log_probs, top_scores, top_eos_flags = \
            tf.cond(tf.equal(current_time_step, 1),
                    lambda: [init_top_sequences, init_top_log_probs, init_top_scores, init_top_eos_flags],
                    lambda: [top_sequences, top_log_probs, top_scores, top_eos_flags])

        return top_sequences, top_log_probs, top_scores, top_eos_flags, top_memories

    def _update_alive(top_sequences, top_scores, top_log_probs, top_eos_flags, top_memories):
        """ Assembles an updated set of unfinished beam candidates from the set of top-k translation hypotheses
        generated at the current time-step; top-k for the incoming sequences in 2 * beam_size """
        # Exclude completed sequences from the alive beam by setting their scores to a large negative value
        top_scores += tf.to_float(top_eos_flags) * (-1. * 1e7)
        # Update the alive beam
        updated_alive_sequences, updated_alive_log_probs, updated_alive_eos_flags, updated_alive_memories = \
            gather_top_sequences(top_sequences,
                                 top_scores,
                                 top_log_probs,
                                 top_eos_flags,
                                 top_memories,
                                 beam_size,
                                 batch_size,
                                 'alive')

        return updated_alive_sequences, updated_alive_log_probs, updated_alive_eos_flags, updated_alive_memories

    def _update_finished(finished_sequences, finished_scores, finished_eos_flags, top_sequences, top_scores,
                         top_eos_flags):
        """ Updates the list of completed translation hypotheses (i.e. ones terminating in <EOS>) on the basis of the
        top-k hypotheses generated at the current time-step; top-k for the incoming sequences in 2 * beam_size """
        # Match the length of the 'finished sequences' tensor with the length of the 'finished scores' tensor
        zero_padding = tf.zeros([batch_size, beam_size, 1], dtype=int_dtype)
        finished_sequences = tf.concat([finished_sequences, zero_padding], axis=2)
        # Exclude incomplete sequences from the finished beam by setting their scores to a large negative value
        top_scores += (1. - tf.to_float(top_eos_flags)) * (-1. * 1e7)
        # Combine sequences finished at previous time steps with the top sequences from current time step, as well as
        # their scores and eos-flags, for the selection of a new, most likely, set of finished sequences
        top_finished_sequences = tf.concat([finished_sequences, top_sequences], axis=1)
        top_finished_scores = tf.concat([finished_scores, top_scores], axis=1)
        top_finished_eos_flags = tf.concat([finished_eos_flags, top_eos_flags], axis=1)
        # Update the finished beam
        updated_finished_sequences, updated_finished_scores, updated_finished_eos_flags, _ = \
            gather_top_sequences(top_finished_sequences,
                                 top_finished_scores,
                                 top_finished_scores,
                                 top_finished_eos_flags,
                                 None,
                                 beam_size,
                                 batch_size,
                                 'finished')

        return updated_finished_sequences, updated_finished_scores, updated_finished_eos_flags

    def _decoding_step(current_time_step,
                       alive_sequences,
                       alive_log_probs,
                       finished_sequences,
                       finished_scores,
                       finished_eos_flags,
                       alive_memories):
        """ Defines a single step of greedy decoding. """
        # 1. Get the top sequences/ scores/ flags for the current time step
        top_sequences, top_log_probs, top_scores, top_eos_flags, top_memories = \
            _extend_hypotheses(current_time_step,
                               alive_sequences,
                               alive_log_probs,
                               alive_memories)

        # 2. Update the alive beam
        alive_sequences, alive_log_probs, alive_eos_flags, alive_memories = \
            _update_alive(top_sequences,
                          top_scores,
                          top_log_probs,
                          top_eos_flags,
                          top_memories)

        # 3. Update the finished beam
        finished_sequences, finished_scores, finished_eos_flags = \
            _update_finished(finished_sequences,
                             finished_scores,
                             finished_eos_flags,
                             top_sequences,
                             top_scores,
                             top_eos_flags)

        return current_time_step + 1, alive_sequences, alive_log_probs, finished_sequences, finished_scores, \
               finished_eos_flags, alive_memories

    def _continue_decoding(curr_time_step,
                           alive_sequences,
                           alive_log_probs,
                           finished_sequences,
                           finished_scores,
                           finished_eos_flags,
                           alive_memories):
        """ Returns 'True' if all of the sequences in the extended hypotheses exceeded the maximum specified
        length or if none of the extended hypotheses are more likely than the lowest scoring finished hypothesis. """
        # Check if the maximum prediction length has been reached
        length_criterion = tf.greater(curr_time_step, translation_maxlen)

        # Otherwise, check if the most likely alive hypothesis is less likely than the least probable completed sequence
        # Calculate the best possible score of the most probably sequence currently alive
        max_length_penalty = 1.
        if normalization_alpha > 0.:
            max_length_penalty = ((5. + tf.to_float(translation_maxlen)) ** normalization_alpha) / \
                                 ((5. + 1.) ** normalization_alpha)

        highest_alive_score = alive_log_probs[:, 0] / max_length_penalty
        # Calculate the score of the least likely sequence currently finished
        lowest_finished_score = tf.reduce_min(finished_scores * tf.cast(finished_eos_flags, float_dtype), axis=1)
        # Account for the case in which none of the sequences in 'finished' have terminated so far;
        # In that case, each of the unfinished sequences is assigned a high negative probability, so that the
        # termination condition is not met
        mask_unfinished = (1. - tf.to_float(tf.reduce_any(finished_eos_flags, 1))) * (-1. * 1e7)
        lowest_finished_score += mask_unfinished

        # Check is the current highest alive score is lower than the current lowest finished score
        likelihood_criterion = tf.reduce_all(tf.greater(lowest_finished_score, highest_alive_score))

        # Decide whether to continue the decoding process
        return tf.logical_not(tf.logical_or(length_criterion, likelihood_criterion))

    # Initialize alive sequence and score trackers and expand to beam size
    alive_log_probs = tf.zeros([batch_size, beam_size])

    # Initialize decoded sequences
    alive_sequences = tf.expand_dims(batch_to_beam(initial_ids, beam_size), 2)

    # Initialize finished sequence, score, and flag trackers
    finished_sequences = tf.expand_dims(batch_to_beam(initial_ids, beam_size), 2)
    finished_scores = tf.ones([batch_size, beam_size]) * (-1. * 1e7)  # initialize to a low value
    finished_eos_flags = tf.zeros([batch_size, beam_size], dtype=tf.bool)

    # Initialize memories
    alive_memories = initial_memories

    # Execute the auto-regressive decoding step via while loop
    _, alive_sequences, alive_log_probs, finished_sequences, finished_scores, finished_eos_flags, _ = \
        tf.while_loop(
            _continue_decoding,
            _decoding_step,
            [tf.constant(1), alive_sequences, alive_log_probs, finished_sequences, finished_scores, finished_eos_flags,
             alive_memories],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None, None]),
                              alive_log_probs.get_shape(),
                              tf.TensorShape([None, None, None]),
                              finished_scores.get_shape(),
                              finished_eos_flags.get_shape(),
                              get_memory_invariants(alive_memories)],
            parallel_iterations=10,
            swap_memory=False,
            back_prop=False)

    alive_sequences.set_shape((None, beam_size, None))
    finished_sequences.set_shape((None, beam_size, None))

    # Account for the case in which a particular sequence never terminates in <EOS>;
    # in that case, copy the contents of the alive beam for that item into the finished beam (sequence + score)
    # tf.reduce_any(finished_eos_flags, 1) is False if there exists no completed translation hypothesis for a source
    # sentence in either of the beams , i.e. no replacement takes place if there is at least one finished translation
    finished_sequences = tf.where(tf.reduce_any(finished_eos_flags, 1), finished_sequences, alive_sequences)
    # Attention: alive_scores are not length normalized!
    finished_scores = tf.where(tf.reduce_any(finished_eos_flags, 1), finished_scores, alive_log_probs)
    # Truncate initial <GO> in finished sequences
    finished_sequences = finished_sequences[:, :, 1:]

    return finished_sequences, finished_scores
