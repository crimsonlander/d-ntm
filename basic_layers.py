import tensorflow as tf

from abc import ABCMeta, abstractmethod
from helpers import conditional_reset, NameCreator, class_with_name_scope, function_with_name_scope


@class_with_name_scope
class BaseLayer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def save_state(self):
        """
        :return: Graph operation to write current state to saved state.
        """

    @abstractmethod
    def feed_input(self, i):
        """
        Construct layer output tensor given input tensor and update layer inner state tensors accordingly.
        :param i: Input tensor.
        :return: Output tensor.
        """

    @abstractmethod
    def reset_saved_state(self):
        """
        :return: Graph operation to reset saved state to initial value.
        """

    @abstractmethod
    def reset_current_state(self):
        """
        Replace current state tensor with default values.
        :return: None
        """
    @abstractmethod
    def reset_current_state_if(self, cond):
        """
        Replace current state tensor with default values if cond is True.
        :param cond: scalar boolean tensor
        :return: None
        """

    def feed_sequence(self, seq):
        """
        :param seq: Python sequence of input tensors.
        :return: List of output tensors.
        """
        outputs = []
        for i in seq:
            outputs.append(self.feed_input(i))

        return outputs

    def feed_sequence_tensor(self, seq_tensor, seq_len):
        seq = map(lambda w: tf.squeeze(w, [1]), tf.split(1, seq_len, seq_tensor))
        return self.feed_sequence(seq)

    def feed_sequence_tensor_embeddings(self, seq_tensor, seq_len, embeddings):
        """
        :param seq_tensor: Tensor of shape (batch_size, seq_len, input_dim1, ..., input dimN) with integer values
            in range [0, num_classes).
        :param seq_len: Python integer (not tensor), should be equal to dimension 1 of seq_tensor.
        :param embeddings: Embeddings tensor for embedding_lookup.
        :return: List of output tensors.
        """
        seq_embeddings = tf.nn.embedding_lookup(embeddings, seq_tensor)
        embeddings_sequence = map(lambda w: tf.squeeze(w, squeeze_dims=[1]),
                                  tf.split(1, seq_len, seq_embeddings))
        return self.feed_sequence(embeddings_sequence)


@class_with_name_scope
class LSTM(BaseLayer):
    def __init__(self, input_size, output_size, batch_size, name=None, init_weights=True):
        self.name = NameCreator.name_it(self, name)
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self._state_shape = [batch_size, output_size]

        with tf.variable_scope(self.name):
            self.saved_output = tf.Variable(tf.zeros(self._state_shape),
                                            trainable=False, name='saved_output')
            self.saved_state = tf.Variable(tf.zeros(self._state_shape),
                                           trainable=False, name='saved_input')

        self.output = self.saved_output
        self.state = self.saved_state

        if init_weights:
            with tf.variable_scope(self.name):
                self.iW = tf.Variable(tf.truncated_normal([input_size, 4 * output_size], -0.1, 0.1), name='iW')
                self.oW = tf.Variable(tf.truncated_normal([output_size, 4 * output_size], -0.1, 0.1), name='oW')
                self.b = tf.Variable(tf.zeros([1, 4 * output_size]), name='b')

            self.eval_model = type(self)(input_size, output_size, 1,
                                         name=self.name + '_eval', init_weights=False)
            self.eval_model.iW = self.iW
            self.eval_model.oW = self.oW
            self.eval_model.b = self.b

    def save_state(self):
        return tf.group(
            self.saved_output.assign(self.output),
            self.saved_state.assign(self.state))

    def feed_input(self, i):
        v = tf.matmul(i, self.iW) + tf.matmul(self.output, self.oW) + self.b
        input_gate, forget_gate, update, output_gate = tf.split(1, 4, v)
        self.state = tf.sigmoid(forget_gate) * self.state + tf.sigmoid(input_gate) * tf.tanh(update)
        self.output = tf.sigmoid(output_gate) * tf.tanh(self.state)

        return self.output

    def reset_saved_state(self):
        with tf.variable_scope(self.name):
            return tf.group(
                self.saved_output.assign(tf.zeros([self.batch_size, self.output_size])),
                self.saved_state.assign(tf.zeros([self.batch_size, self.output_size])))

    def reset_current_state(self):
        with tf.variable_scope(self.name):
            self.state = tf.zeros(self._state_shape)
            self.output = tf.zeros(self._state_shape)

    def reset_current_state_if(self, cond):
        with tf.variable_scope(self.name):
            self.state = conditional_reset(self.state, self._state_shape, cond)
            self.output = conditional_reset(self.output, self._state_shape, cond)


class SparseLSTM(LSTM):
    def __init__(self, num_classes, output_size, batch_size, name=None, init_weights=True):
        LSTM.__init__(self, num_classes, output_size, batch_size, name=name, init_weights=init_weights)
        self.input_size = None
        self.num_classes = num_classes

    @function_with_name_scope
    def feed_input(self, i):
        v = tf.nn.embedding_lookup(self.iW, i) + tf.matmul(self.output, self.oW) + self.b
        input_gate, forget_gate, update, output_gate = tf.split(1, 4, v)
        self.state = tf.sigmoid(forget_gate) * self.state + tf.sigmoid(input_gate) * tf.tanh(update)
        self.output = tf.sigmoid(output_gate) * tf.tanh(self.state)

        return self.output

    def feed_sequence_tensor_embeddings(self, seq_tensor, seq_len, embeddings):
        raise NotImplementedError()


@class_with_name_scope
class GRU(BaseLayer):
    def __init__(self, input_size, output_size, batch_size, name=None, init_weights=True):
        self.name = NameCreator.name_it(self, name)
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self._state_shape = [batch_size, output_size]

        with tf.variable_scope(self.name):
            if init_weights:
                self.iW_g = tf.Variable(
                    tf.truncated_normal([input_size, 2 * output_size], -0.1, 0.1), name='iW_g')
                self.oW_g = tf.Variable(
                    tf.truncated_normal([output_size, 2 * output_size], -0.1, 0.1), name='oW_g')

                self.iW = tf.Variable(
                    tf.truncated_normal([input_size, output_size], -0.1, 0.1), name='iW')
                self.oW = tf.Variable(
                    tf.truncated_normal([output_size, output_size], -0.1, 0.1), name='oW')

                self.b_g = tf.Variable(tf.zeros([1, 2 * output_size]), name='b_g')
                self.b = tf.Variable(tf.zeros([1, output_size]), name='b')

            self.saved_output = tf.Variable(
                tf.zeros(self._state_shape), trainable=False, name='saved_output')
            self.output = self.saved_output

        if init_weights:
            self.eval_model = type(self)(input_size, output_size, 1,
                                         name=self.name + '_eval', init_weights=False)
            self.eval_model.iW_g = self.iW_g
            self.eval_model.oW_g = self.oW_g
            self.eval_model.iW = self.iW
            self.eval_model.oW = self.oW
            self.eval_model.b_g = self.b_g
            self.eval_model.b = self.b

    def save_state(self):
        return self.saved_output.assign(self.output)

    def feed_input(self, i):
        g = tf.nn.sigmoid(tf.matmul(i, self.iW_g) + tf.matmul(self.output, self.oW_g) + self.b_g)
        u, r = tf.split(1, 2, g)
        h = tf.nn.tanh(tf.matmul(i, self.iW) + r * tf.matmul(self.output, self.oW) + self.b)
        self.output = u * h + (1. - u) * self.output

        return self.output

    def reset_saved_state(self):
        return self.saved_output.assign(tf.zeros(self._state_shape))

    def reset_current_state(self):
        self.output = tf.zeros(self._state_shape)

    def reset_current_state_if(self, cond):
        self.output = conditional_reset(self.output, self._state_shape, cond)


class SparseGRU(GRU):
    def __init__(self, num_classes, output_size, batch_size, name=None, init_weights=True):
        GRU.__init__(self, num_classes, output_size, batch_size, name=name, init_weights=init_weights)
        self.input_size = None
        self.num_classes = num_classes

    @function_with_name_scope
    def feed_input(self, i):
        g = tf.nn.sigmoid(tf.nn.embedding_lookup(self.iW_g, i) + tf.matmul(self.output, self.oW_g) + self.b_g)
        u, r = tf.split(1, 2, g)
        h = tf.nn.tanh(tf.nn.embedding_lookup(self.iW, i) + r * tf.matmul(self.output, self.oW) + self.b)
        self.output = u * h + (1. - u) * self.output
        return self.output

    def feed_sequence_tensor_embeddings(self, seq_tensor, seq_len, embeddings):
        raise NotImplementedError()


@class_with_name_scope
class RNN(BaseLayer):
    def __init__(self, input_size, output_size, batch_size,
                 activation_function=tf.nn.sigmoid, name=None, init_weights=True):
        self.name = NameCreator.name_it(self, name)
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.activation_function = activation_function
        self._state_shape = [batch_size, output_size]

        with tf.variable_scope(self.name):
            if init_weights:
                self.iW = tf.Variable(tf.truncated_normal([input_size, output_size], -0.1, 0.1))
                self.oW = tf.Variable(tf.truncated_normal([output_size, output_size], -0.1, 0.1))
                self.vb = tf.Variable(tf.zeros([1, output_size]))

            self.saved_output = tf.Variable(tf.zeros(self._state_shape), trainable=False)

            self.output = self.saved_output

        if init_weights:
            self.eval_model = type(self)(input_size, output_size, 1, activation_function,
                                         self.name + '_eval', False)
            self.eval_model.iW = self.iW
            self.eval_model.oW = self.oW
            self.eval_model.vb = self.vb

    def save_state(self):
        return self.saved_output.assign(self.output)

    def feed_input(self, i):
        self.output = self.activation_function(
            tf.matmul(i, self.iW) + tf.matmul(self.output, self.oW) + self.vb)
        return self.output

    def reset_saved_state(self):
        return self.saved_output.assign(tf.zeros(self._state_shape))

    def reset_current_state(self):
        self.output = tf.zeros(self._state_shape)

    def reset_current_state_if(self, cond):
        self.output = conditional_reset(self.output, self._state_shape, cond)


@class_with_name_scope
class FeedForward(BaseLayer):
    def __init__(self, input_size, output_size, batch_size,
                 activation_function=None, name=None, init_weights=True):
        self.name = NameCreator.name_it(self, name)
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.activation_function = activation_function

        if init_weights:
            with tf.variable_scope(self.name):
                self.W = tf.Variable(tf.truncated_normal([input_size, output_size], -0.1, 0.1), name='W')
                self.b = tf.Variable(tf.zeros([1, output_size]), name='b')

            self.eval_model = type(self)(input_size, output_size, 1, activation_function,
                                         self.name + '_eval', False)
            self.eval_model.W = self.W
            self.eval_model.b = self.b

    def save_state(self):
        return tf.group()

    def feed_input(self, i):
        output = tf.matmul(i, self.W) + self.b
        if self.activation_function:
            return self.activation_function(output)
        return output

    def reset_saved_state(self):
        return tf.group()

    def reset_current_state(self):
        pass

    def reset_current_state_if(self, cond):
        pass


class ConnectLayers(BaseLayer):
    def __init__(self, layers):
        self.layers = layers
        self.input_size = layers[0].input_size
        self.output_size = layers[-1].output_size
        self.batch_size = layers[0].batch_size

        layer_size = layers[0].output_size

        for layer in layers[1:]:
            assert layer.batch_size == self.batch_size
            assert layer.input_size == layer_size
            layer_size = layer.output_size

        if self.batch_size > 1:
            self.eval_model = type(self)([layer.eval_model for layer in layers])

    def save_state(self):
        return tf.group(*[layer.save_state() for layer in self.layers])

    def feed_input(self, i):
        output = i
        for layer in self.layers:
            output = layer.feed_input(output)
        return output

    def reset_saved_state(self):
        return tf.group(*[layer.reset_saved_state() for layer in self.layers])

    def reset_current_state(self):
        for layer in self.layers:
            layer.reset_current_state()

    def reset_current_state_if(self, cond):
        for layer in self.layers:
            layer.reset_current_state_if(cond)
