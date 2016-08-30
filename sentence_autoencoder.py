import tensorflow as tf
import numpy as np

from six.moves import cPickle as pickle
from babi_batch_generator import bAbiBatchGenerator, num2word
from basic_layers import LSTM, SparseGRU, SparseLSTM, FeedForward, ConnectLayers, BatchNormalization

batch_size = 100

graph = tf.Graph()
gen = bAbiBatchGenerator(1, batch_size)
eval_gen = bAbiBatchGenerator(1, 1, 'test')

sentence_encoding_size = 200
vocabulary_size = gen.vocabulary_size
sentence_padding = gen.padding

summaries_dir = '/tmp/TensorBoard/summaries/bAbi/lstm_autoencoder'

with graph.as_default():
    decoder_nodes = 50

    encoder = SparseGRU(vocabulary_size, sentence_encoding_size, batch_size)

    decoder_lstm = LSTM(sentence_encoding_size, decoder_nodes, batch_size)
    decoder_linear = FeedForward(decoder_nodes, vocabulary_size, batch_size)
    decoder = ConnectLayers((decoder_lstm, decoder_linear))

    prediction_length = sentence_padding

    blank_input = tf.zeros((batch_size, sentence_encoding_size))

    with tf.variable_scope("training"):

        fact = tf.placeholder(tf.int32, (batch_size, sentence_padding), name="input")
        mask = tf.cast(tf.not_equal(fact, 0), tf.float32)
        logits = []
        encoder_outputs = encoder.feed_sequence_tensor(fact, sentence_padding)
        encoding = encoder_outputs[-1]
        #encoding = batch_norm.feed_input(encoding)
        losses = []

        for i in range(prediction_length):
            logits.append(decoder.feed_input(encoding))

            index = sentence_padding - prediction_length + i
            losses.append(mask[:, index] * tf.nn.sparse_softmax_cross_entropy_with_logits(logits[i], fact[:, index]))

        #with tf.control_dependencies([batch_norm.save_state()]):
        loss = tf.reduce_mean(tf.concat(0, losses))
        optimize = tf.train.AdadeltaOptimizer(0.1).minimize(loss)

        grads = tf.gradients(loss, encoder_outputs)
        for i in range(len(grads)):
            if grads[i] is not None:
                tf.histogram_summary("encoder output gradients %d" % i, grads[i])


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

    summaries = tf.merge_all_summaries()

train_writer = tf.train.SummaryWriter(summaries_dir + '/train')

num_steps = 20000
summary_frequency = 100
valid_size = 10

if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir + '/train')

with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    #encoder_saved_weights = pickle.load(open("encoder.save", "rb"))
    #encoder.set_weights(encoder_saved_weights)
    train_writer.add_graph(graph)
    for step in range(num_steps):
        train_feed = dict()
        sentences, _, _, _ = gen.get_next_batch()
        train_feed[fact] = sentences

        if step % summary_frequency == summary_frequency - 1:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            l, _, summary = sess.run([loss, optimize, summaries],
                                     feed_dict=train_feed,
                                     options=run_options,
                                     run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            train_writer.add_summary(summary, step)

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
                print("%40s | %40s" % (q_string, decoded))

            print("Validation loss:", valid_loss_acc / valid_size)
            print('#' * 80)

        elif step % 10 == 9:
            l, _, summary = sess.run([loss, optimize, summaries], feed_dict=train_feed)
            train_writer.add_summary(summary, step)

        else:
            l, _ = sess.run([loss, optimize], feed_dict=train_feed)

    weights = encoder.get_weights()
    pickle.dump(weights, open("encoder.save", "wb"))
    train_writer.close()



