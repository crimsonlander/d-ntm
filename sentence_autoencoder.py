import tensorflow as tf
import numpy as np

from babi_batch_generator import bAbiBatchGenerator, num2word
from basic_layers import LSTM, SparseGRU, FeedForward, ConnectLayers, BatchNormalization

batch_size = 100

graph = tf.Graph()
gen = bAbiBatchGenerator(1, batch_size)
eval_gen = bAbiBatchGenerator(1, 1, 'test')

sentence_encoding_size = 300
vocabulary_size = gen.vocabulary_size
sentence_padding = gen.padding
num_unrollings = 15

with graph.as_default():
    #batch_norm = BatchNormalization(sentence_encoding_size, batch_size, std_epsilon=1e-2, scale=1.)
    encoder = SparseGRU(vocabulary_size, sentence_encoding_size, batch_size)
    lstm = LSTM(sentence_encoding_size, 100, batch_size)
    linear = FeedForward(100, vocabulary_size, batch_size)
    decoder = ConnectLayers((lstm, linear))

    prediction_length = 1

    with tf.variable_scope("training"):

        fact = tf.placeholder(tf.int32, (batch_size, sentence_padding), name="input")
        logits = []
        encoding = encoder.feed_sequence_tensor(fact, sentence_padding)[-1]
        #encoding = batch_norm.feed_input(encoding)
        losses = []
        for i in range(prediction_length):
            logits.append(decoder.feed_input(encoding))
            losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits[i], fact[:, i]))

        #with tf.control_dependencies([batch_norm.save_state()]):
        loss = tf.reduce_mean(tf.concat(0, losses))
        optimize = tf.train.AdadeltaOptimizer(0.01).minimize(loss)

    with tf.variable_scope("validation"):
        valid_input = tf.placeholder(tf.int32, (1, sentence_padding), name="input")
        valid_logits = []
        valid_encoding = encoder.eval_model.feed_sequence_tensor(valid_input, sentence_padding)[-1]
        #valid_encoding = batch_norm.eval_model.feed_input(valid_encoding)
        valid_losses = []
        valid_predictions = []
        for i in range(sentence_padding):
            valid_logits.append(decoder.eval_model.feed_input(valid_encoding))
            valid_losses.append(tf.nn.sparse_softmax_cross_entropy_with_logits(valid_logits[i], valid_input[:, i]))

        valid_loss = tf.reduce_mean(tf.concat(0, valid_losses))
        for i in range(len(valid_logits)):
            valid_predictions.append(tf.reshape(tf.arg_max(valid_logits[i], 1), (1, 1)))
        valid_prediction = tf.concat(1, valid_predictions)

num_steps = 20000
summary_frequency = 100
valid_size = 10

with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    for step in range(num_steps):
        train_feed = dict()
        sentences, _, _, _ = gen.get_next_batch()
        train_feed[fact] = sentences
        l, _ = sess.run([loss, optimize], feed_dict=train_feed)

        if step % summary_frequency == summary_frequency - 1:
            print(step, l)
            valid_loss_acc = 0.
            for i in range(valid_size):
                f, a, q, s = eval_gen.get_next_batch()
                q_string, a_string = eval_gen.batch2strings((f, a, q, s))[0]
                valid_feed = dict()
                valid_feed[valid_input] = f
                l, p = sess.run([valid_loss, valid_prediction], feed_dict=valid_feed)
                valid_loss_acc += l
                decoded, _ = eval_gen.batch2strings((p, a, q, s))[0]
                print("%35s | %35s" % (q_string, decoded))

            print("Validation loss:", valid_loss_acc / valid_size)
            print('#' * 80)