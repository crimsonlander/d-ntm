import tensorflow as tf

from abc import ABCMeta, abstractmethod
from helpers import conditional_reset, NameCreator, class_with_name_scope, function_with_name_scope, function_args


class BaseLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, input_size, output_size, batch_size, name, copy_from, *args):
        if type(self).__name__ == "BaseLayer":
            raise NotImplementedError("abstract class")
        self.name = NameCreator.name_it(self, name)
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

        if not hasattr(self, "weight_variables"):
            self.weight_variables = dict()

        if not hasattr(self, "state_variables"):
            self.state_variables = dict()

        if not hasattr(self, "layers"):
            self.layers = []

        if copy_from is None:
            for k, v in self.weight_variables.items():
                setattr(self, k, tf.Variable(v, name=k))
        else:
            self.weight_variables = copy_from.weight_variables
            for k, v in self.weight_variables.items():
                setattr(self, k, getattr(copy_from, k))

        self.current_state_variable_names = []
        self.saved_state_variables = []
        self.state_default_values = []
        for k, v in self.state_variables.items():
            var = tf.Variable(v, trainable=False, name='saved_' + k)
            setattr(self, "saved_" + k, var)
            setattr(self, k, var)
            self.current_state_variable_names.append(k)
            self.saved_state_variables.append(var)
            self.state_default_values.append(v)

    def save_state(self):
        """
        :return: Graph operation to write current state to saved state.
        """
        save_layers = [layer.save_state() for layer in self.layers]
        save_own_state = []

        for i in range(len(self.saved_state_variables)):
            current_state = getattr(self, self.current_state_variable_names[i])
            save_own_state.append(self.saved_state_variables[i].assign(current_state))

        return tf.group(*(save_layers + save_own_state))

    @abstractmethod
    def feed_input(self, i):
        """
        Construct layer output tensor given input tensor and update layer inner state tensors accordingly.
        :param i: Input tensor.
        :return: Output tensor.
        """

    def reset_saved_state(self):
        """
        :return: Graph operation to reset saved state to initial value.
        """
        reset_layers = [layer.reset_saved_state() for layer in self.layers]
        reset_own_state = []

        for i in range(len(self.saved_state_variables)):
            default_state = self.state_default_values[i]
            reset_own_state.append(self.saved_state_variables[i].assign(default_state))

        return tf.group(*(reset_layers + reset_own_state))

    def reset_current_state(self):
        """
        Replace current state tensor with default values.
        :return: None
        """
        for layer in self.layers:
            layer.reset_current_state()

        for i in range(len(self.current_state_variable_names)):
            setattr(self, self.current_state_variable_names[i], self.state_default_values[i])

    def reset_current_state_if(self, cond):
        """
        Replace current state tensor with default values if cond is True.
        :param cond: scalar boolean tensor
        :return: None
        """
        for layer in self.layers:
            layer.reset_current_state_if(cond)

        for i in range(len(self.current_state_variable_names)):
            current = getattr(self, self.current_state_variable_names[i])
            setattr(self, self.current_state_variable_names[i],
                    conditional_reset(current, self.state_default_values[i], cond))

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
        """
        Split seq_tensor along dimension 1 and feed resulting sequence. Dimension 1 will be squeezed.
        """
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
    def __init__(self, input_size, output_size, batch_size, name=None, copy_from=None):
        self._state_shape = [batch_size, output_size]
        self.name = NameCreator.name_it(self, name)
        with tf.variable_scope(self.name):
            self.state_variables = {"state": tf.zeros(self._state_shape, name="default_state"),
                                    "output": tf.zeros(self._state_shape, name="default_output")}

            self.weight_variables = {"iW": tf.truncated_normal([input_size, 4 * output_size], -0.1, 0.1),
                                     "oW": tf.truncated_normal([output_size, 4 * output_size], -0.1, 0.1),
                                     "b": tf.zeros([1, 4 * output_size])}
            BaseLayer.__init__(*function_args())
            if copy_from is None:
                self.eval_model = type(self)(input_size, output_size, 1, self.name + '_eval', self)

    def feed_input(self, i):
        v = tf.matmul(i, self.iW) + tf.matmul(self.output, self.oW) + self.b
        input_gate, forget_gate, update, output_gate = tf.split(1, 4, v)
        self.state = tf.sigmoid(forget_gate) * self.state + tf.sigmoid(input_gate) * tf.tanh(update)
        self.output = tf.sigmoid(output_gate) * tf.tanh(self.state)

        return self.output


class SparseLSTM(LSTM):
    def __init__(self, num_classes, output_size, batch_size, name=None, copy_from=None):
        LSTM.__init__(*function_args())
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
    def __init__(self, input_size, output_size, batch_size, name=None, copy_from=None):
        self._state_shape = [batch_size, output_size]
        self.name = NameCreator.name_it(self, name)
        with tf.variable_scope(self.name):
            self.state_variables = {"output": tf.zeros(self._state_shape, name="default_output")}

            self.weight_variables = {"iW_g": tf.truncated_normal([input_size, 2 * output_size], -0.1, 0.1),
                                     "oW_g": tf.truncated_normal([output_size, 2 * output_size], -0.1, 0.1),
                                     "iW": tf.truncated_normal([input_size, output_size], -0.1, 0.1),
                                     "oW": tf.truncated_normal([output_size, output_size], -0.1, 0.1),

                                     "b_g": tf.zeros([1, 2 * output_size]),
                                     "b": tf.zeros([1, output_size])}
            BaseLayer.__init__(*function_args())
            if copy_from is None:
                self.eval_model = type(self)(input_size, output_size, 1, self.name + '_eval', self)

    def feed_input(self, i):
        g = tf.nn.sigmoid(tf.matmul(i, self.iW_g) + tf.matmul(self.output, self.oW_g) + self.b_g)
        u, r = tf.split(1, 2, g)
        h = tf.nn.tanh(tf.matmul(i, self.iW) + r * tf.matmul(self.output, self.oW) + self.b)
        self.output = u * h + (1. - u) * self.output

        return self.output


class SparseGRU(GRU):
    def __init__(self, num_classes, output_size, batch_size, name=None, copy_from=None):
        GRU.__init__(*function_args())
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
                 name=None, copy_from=None, activation_function=tf.nn.sigmoid):
        self._state_shape = [batch_size, output_size]
        self.name = NameCreator.name_it(self, name)
        self.activation_function = activation_function
        with tf.variable_scope(self.name):
            self.state_variables = {"output": tf.zeros(self._state_shape, name="default_output")}

            self.weight_variables = {"iW": tf.truncated_normal([input_size, output_size], -0.1, 0.1),
                                     "oW": tf.truncated_normal([output_size, output_size], -0.1, 0.1),
                                     "b": tf.zeros([1, output_size])}
            BaseLayer.__init__(*function_args())
            if copy_from is None:
                self.eval_model = type(self)(input_size, output_size, 1, self.name + '_eval', self, activation_function)

    def feed_input(self, i):
        self.output = self.activation_function(
            tf.matmul(i, self.iW) + tf.matmul(self.output, self.oW) + self.vb)
        return self.output


@class_with_name_scope
class FeedForward(BaseLayer):
    def __init__(self, input_size, output_size, batch_size,
                 name=None, copy_from=None, activation_function=None):
        self.name = NameCreator.name_it(self, name)
        self.activation_function = activation_function
        with tf.variable_scope(self.name):
            self.weight_variables = {"W": tf.truncated_normal([input_size, output_size], -0.1, 0.1),
                                     "b": tf.zeros([1, output_size])}
            BaseLayer.__init__(*function_args())
            if copy_from is None:
                self.eval_model = type(self)(input_size, output_size, 1, self.name + '_eval', self, activation_function)

    def feed_input(self, i):
        output = tf.matmul(i, self.W) + self.b
        if self.activation_function:
            return self.activation_function(output)
        return output


class ConnectLayers(BaseLayer):
    def __init__(self, layers, name=None, copy_from=None):
        if copy_from is None:
            self.layers = layers
        else:
            self.layers = [layer.eval_model for layer in copy_from.layers]

        batch_size = self.layers[0].batch_size
        layer_size = self.layers[0].output_size

        for layer in self.layers[1:]:
            assert layer.batch_size == batch_size
            assert layer.input_size == layer_size
            layer_size = layer.output_size

        BaseLayer.__init__(self, self.layers[0].input_size, self.layers[-1].output_size, batch_size, name, copy_from)
        if copy_from is None:
            self.eval_model = type(self)(None, self.name + "_eval", self)

    def feed_input(self, i):
        output = i
        for layer in self.layers:
            output = layer.feed_input(output)
        return output


@class_with_name_scope
class BatchNormalization(BaseLayer):
    def __init__(self, input_size, batch_size, std_epsilon=1e-4, scale=1., shift=0., name=None, copy_from=None):
        self.std_epsilon = std_epsilon
        self.scale = scale
        self.shift = shift
        self._shape = (1, input_size)
        self.state_variables = {"mean": tf.zeros(self._shape),
                                "std": tf.ones(self._shape)}
        BaseLayer.__init__(self, input_size, input_size, batch_size, name, copy_from)
        if copy_from is None:
            self.update = True
            self.eval_model = type(self)(input_size, 1, std_epsilon, scale, shift, self.name + "_eval", self)
            self.eval_model.saved_mean = self.saved_mean
            self.eval_model.saved_std = self.saved_std
        else:
            self.update = False

        if self.update:
            self.inputs = []

    def feed_input(self, i):
        if self.update:
            self.inputs.append(i)

        return (i - self.saved_mean) / (self.saved_std + self.std_epsilon) * self.scale + self.shift

    def save_state(self):
        if not self.update:
            return tf.group()

        concat_inputs = tf.concat(0, self.inputs)
        i_mean = tf.reduce_mean(concat_inputs, [0])
        i_std = tf.reduce_mean((concat_inputs - i_mean) ** 2, [0])

        return tf.group(self.saved_mean.assign(self.saved_mean * 0.95 + i_mean * 0.05),
                        self.saved_std.assign(self.saved_std * 0.95 + i_std * 0.05))

    def reset_saved_state(self):
        pass

    def reset_current_state(self):
        pass

    def reset_current_state_if(self, cond):
        pass