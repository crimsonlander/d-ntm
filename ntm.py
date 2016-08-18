import tensorflow as tf
from basic_layers import BaseLayer, FeedForward, ConnectLayers
from helpers import class_with_name_scope, conditional_reset, NameCreator


@class_with_name_scope
class Tape(object):
    def __init__(self, mem_key_size, mem_content_size, num_cells, batch_size, name=None, init_weights=True):
        self.name = NameCreator.name_it(self, name)
        self.mem_key_size = mem_key_size
        self.mem_content_size = mem_content_size
        self.num_cells = num_cells
        self.batch_size = batch_size
        self._content_size = [batch_size, num_cells, mem_content_size]

        with tf.variable_scope(self.name):
            if init_weights:
                # Additional key for NOP
                self.keys = tf.Variable(tf.truncated_normal([num_cells + 1, mem_key_size], -0.1, 0.1), name='keys')

            self.saved_content = tf.Variable(tf.zeros(self._content_size),
                                             dtype=tf.float32, trainable=False, name='keys')
            self.content = self.saved_content

        if init_weights:
            self.eval_model = Tape(mem_key_size, mem_content_size, num_cells, 1, self.name + '_eval', False)
            self.eval_model.keys = self.keys

    def _lookup(self, keys, key_strengths):
        """
        Content-based addressing for a batch of keys.

        :param keys: (self.batch_size, self.mem_key_size)-matrix, each row is a key.
        :param key_strengths: (self.batch_size, 1)-matrix, each row is a single scalar - key strength.

        :return: (self.batch_size, self.num_cells)-matrix, each row is a soft address.
        """
        keys_norm = tf.reduce_sum(self.keys * self.keys, reduction_indices=[1], keep_dims=True)
        batch_keys_norm = tf.reduce_sum(keys * keys, reduction_indices=[1], keep_dims=True)
        normalization = tf.sqrt(tf.matmul(batch_keys_norm, keys_norm, transpose_b=True))
        similarity = tf.matmul(keys, self.keys, transpose_b=True) / normalization
        address_weights = tf.nn.softmax(key_strengths * similarity)

        # Don't return NOP weight
        return address_weights[:, 1:]

    def read(self, keys, key_strengths):
        """
        Read batch of values associated with batch of keys using soft addressing scheme.

        :param keys: (self.batch_size, self.mem_key_size)-matrix, each row is a key.
        :param key_strengths: (self.batch_size, 1)-matrix, each row is a single scalar - key strength.

        :return: (self.batch_size, self.mem_content_size)-matrix, each row is a content vector, corresponding to key.
        """
        address_expansion = tf.expand_dims(self._lookup(keys, key_strengths), 2)
        return tf.reduce_sum(address_expansion * self.content, reduction_indices=[1])

    def write(self, keys, key_strengths, erase_gates, content_vectors):
        """
        Write batch of values to cells corresponding to batch of keys using soft addressing scheme.
        Writing to first cell does nothing (NOP).

        :param keys: (self.batch_size, self.mem_key_size)-matrix, each row is a key.
        :param key_strengths: (self.batch_size, 1)-matrix, each row is a single scalar - key strength.
        :param erase_gates: (self.batch_size, 1)-matrix, each row is a single scalar - erase gate.
        Erase vector is used to switch between adding to and replacing old content.
        :param content_vectors: (self.batch_size, self.mem_content_size)-matrix, each row is a content vector.

        :return: Nothing.
        """
        # First content cell is ignored
        address_expansion = tf.expand_dims(self._lookup(keys, key_strengths), 2)
        content_expansion = tf.expand_dims(content_vectors, 1)
        erase_expansion = tf.expand_dims(erase_gates, 1)

        self.content = (1.0 - erase_expansion * address_expansion) * self.content \
            + content_expansion * address_expansion

    def save_state(self):
        return self.saved_content.assign(self.content)

    def reset_saved_state(self):
        return self.saved_content.assign(tf.zeros(self._content_size))

    def reset_current_state(self):
        self.content = tf.zeros(self._content_size)

    def reset_current_state_if(self, cond):
        self.content = conditional_reset(self.content, self._content_size, cond)


@class_with_name_scope
class NTM(BaseLayer):
    def __init__(self, input_size, output_size, batch_size,
                 controller, mem_key_size, mem_content_size, num_cells,
                 activation_function=None, name=None, init_weights=True):

        self.name = NameCreator.name_it(self, name)
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.mem_key_size = mem_key_size
        self.mem_content_size = mem_content_size
        self.num_cells = num_cells
        self.activation_function = activation_function
        self.extended_input_size = input_size + mem_content_size
        self.extended_output_size = output_size + mem_content_size + 2 * mem_key_size + 3
        self._read_result_shape = (batch_size, mem_content_size)

        if init_weights:
            self.tape = Tape(mem_key_size, mem_content_size, num_cells, batch_size, self.name + '_Tape')
            with tf.variable_scope(self.name):
                self.input_adapter = FeedForward(self.extended_input_size, controller.input_size, batch_size)
                self.output_adapter = FeedForward(controller.output_size, self.extended_output_size, batch_size)
                self.controller = controller
                self.extended_controller = ConnectLayers([self.input_adapter, controller, self.output_adapter])

        with tf.variable_scope(self.name):
            self.saved_read_result = tf.Variable(tf.zeros(self._read_result_shape))
            self.read_result = self.saved_read_result

        if batch_size > 1:
            self.eval_model = NTM(input_size, output_size, 1,
                                  controller, mem_key_size, mem_content_size, num_cells,
                                  activation_function, self.name + '_eval', init_weights=False)
            self.eval_model.tape = self.tape.eval_model
            self.eval_model.extended_controller = self.extended_controller.eval_model

    def feed_input(self, i):
        extended_input = tf.concat(1, (i, self.read_result))
        extended_output = self.extended_controller.feed_input(extended_input)

        def slice_output(n):
            result = extended_output[:, slice_output.i:slice_output.i+n]
            slice_output.i += n
            return result
        slice_output.i = 0

        output = slice_output(self.output_size)
        if self.activation_function:
            output = self.activation_function(output)
        write_content = tf.nn.tanh(slice_output(self.mem_content_size))
        write_key = slice_output(self.mem_key_size)
        erase = tf.nn.sigmoid(slice_output(1))
        write_key_str = tf.nn.softplus(slice_output(1))

        read_key = slice_output(self.mem_key_size)
        read_key_str = tf.nn.softplus(slice_output(1))

        self.tape.write(write_key, write_key_str, erase, write_content)
        self.read_result = self.tape.read(read_key, read_key_str)

        return output

    def save_state(self):
        return tf.group(self.controller.save_state,
                        self.tape.save_state(),
                        self.saved_read_result.assign(self.read_result))

    def reset_saved_state(self):
        return tf.group(self.controller.reset_saved_state,
                        self.tape.reset_saved_state(),
                        self.saved_read_result.assign(tf.zeros(self._read_result_shape)))

    def reset_current_state(self):
        self.controller.reset_current_state()
        self.tape.reset_current_state()
        self.read_result = tf.zeros(self._read_result_shape)

    def reset_current_state_if(self, cond):
        self.controller.reset_current_state_if(cond)
        self.tape.reset_current_state_if(cond)
        self.read_result = conditional_reset(self.read_result, self._read_result_shape, cond)
