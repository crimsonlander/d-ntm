import tensorflow as tf
import numpy as np

from babi_batch_generator import bAbiBatchGenerator, num2word
from basic_layers import LSTM, SparseGRU, FeedForward, ConnectLayers, BatchNormalization

batch_size = 32

graph = tf.Graph()
gen = bAbiBatchGenerator(1, batch_size)
eval_gen = bAbiBatchGenerator(1, 1, 'test')

sentence_encoding_size = 200
vocabulary_size = gen.vocabulary_size
sentence_padding = gen.padding
num_unrollings = 15

summaries_dir = '/tmp/TensorBoard/summaries/bAbi/lstm'


def conditional_sparse_softmax_ce_multiple_choice(logits, labels, n_labels, weights, cond, name=None):
    def compute_loss():
        labels_list = list(map(lambda lab: tf.squeeze(lab, [1]), tf.split(1, n_labels, labels)))
        loss = tf.zeros([1])
        for i in range(n_labels):
            loss += weights[i] * tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_list[i]))
        return loss

    return tf.cond(cond, compute_loss, lambda: tf.zeros([1]), name=name)

with graph.as_default():
    batch_norm = BatchNormalization(sentence_encoding_size, batch_size, std_epsilon=1e-2, scale=1.)
    encoder = SparseGRU(vocabulary_size, sentence_encoding_size, batch_size)

    model_layer1 = LSTM(sentence_encoding_size, 200, batch_size)
    model_layer2 = FeedForward(200, vocabulary_size, batch_size)
    model = ConnectLayers((model_layer1, model_layer2))

    input_facts = []
    labels = []
    question_marks = []
    new_story_marks = []
    fact_encodings = []

    losses = []
    predictions = []
    with tf.variable_scope("training"):

        with tf.variable_scope("inputs"):
            for i in range(num_unrollings):
                input_facts.append(
                    tf.placeholder(tf.int32, (batch_size, sentence_padding), name="input_%02d" % i))

        with tf.variable_scope("labels"):
            for i in range(num_unrollings):
                labels.append(tf.placeholder(tf.int32, (batch_size, 2), name="label_%02d" % i))

        with tf.variable_scope("question_marks"):
            for i in range(num_unrollings):
                question_marks.append(tf.placeholder(tf.bool, [], name="question_mark_%02d" % i))

        with tf.variable_scope("new_story_marks"):
            for i in range(num_unrollings):
                new_story_marks.append(tf.placeholder(tf.bool, [], name="new_story_mark_%02d" % i))

        for i in range(num_unrollings):
            fact = input_facts[i]
            is_new_story = new_story_marks[i]
            is_question = question_marks[i]
            label = labels[i]

            encoding = encoder.feed_sequence_tensor(fact, sentence_padding)[-1]
            encoding = batch_norm.feed_input(encoding)
            fact_encodings.append(encoding)
            encoder.reset_current_state()
            model.reset_current_state_if(is_new_story)
            output = model.feed_input(encoding)
            losses.append(conditional_sparse_softmax_ce_multiple_choice(
                output, label, 2, [1, 0], is_question))

        concat_encodings = tf.concat(0, fact_encodings)
        encodings_mean = tf.reduce_mean(concat_encodings, [0], name='encodings_mean')
        encodings_std = tf.sqrt(tf.reduce_mean((concat_encodings - encodings_mean) ** 2, [0]),
                                name='encodings_std')

        tf.histogram_summary('train_encodings_mean', encodings_mean)
        tf.histogram_summary('train_encodings_std', encodings_std)

        loss = tf.reduce_mean(tf.concat(0, losses), name="loss")  # + regularizer

        with tf.control_dependencies([model.save_state(), batch_norm.save_state()]):
            optimize = tf.train.AdadeltaOptimizer(1.).minimize(loss)

    with tf.variable_scope("validation"):
        valid_input = tf.placeholder(tf.int32, (1, sentence_padding), name="input")
        valid_encoding = encoder.eval_model.feed_sequence_tensor(valid_input, sentence_padding)[-1]
        valid_encoding = batch_norm.eval_model.feed_input(valid_encoding)
        encoder.eval_model.reset_current_state()
        valid_is_new_story = tf.placeholder(tf.bool, [], name="new_story_mark")
        model.eval_model.reset_current_state_if(valid_is_new_story)
        valid_output = model.eval_model.feed_input(valid_encoding)
        valid_label = tf.placeholder(tf.int32, (1, 2), name="label")
        valid_is_question = tf.placeholder(tf.bool, [], name="question_mark")

    with tf.control_dependencies([model.eval_model.save_state()]):
        valid_loss = conditional_sparse_softmax_ce_multiple_choice(
            valid_output, valid_label, 2, [1, 0], valid_is_question)

        valid_probs = tf.nn.softmax(valid_output)
        prediction_certainty, valid_prediction = tf.nn.top_k(valid_probs, 2)

    summaries = tf.merge_all_summaries()

train_writer = tf.train.SummaryWriter(summaries_dir + '/train')

num_steps = 5000
summary_frequency = 100
valid_size = 9

if tf.gfile.Exists(summaries_dir):
    tf.gfile.DeleteRecursively(summaries_dir)
tf.gfile.MakeDirs(summaries_dir + '/train')

with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()
    train_writer.add_graph(graph)
    for step in range(num_steps):
        train_feed = dict()
        for j in range(num_unrollings):
            sentences, answers, is_question, is_new_story = gen.get_next_batch()
            train_feed[input_facts[j]] = sentences
            train_feed[labels[j]] = answers
            train_feed[question_marks[j]] = bool(is_question)
            train_feed[new_story_marks[j]] = bool(is_new_story)

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
                valid_feed[valid_label] = a
                valid_feed[valid_is_question] = q
                valid_feed[valid_is_new_story] = s
                l, c, p = sess.run([valid_loss, prediction_certainty, valid_prediction], feed_dict=valid_feed)
                valid_loss_acc += l
                model_answer = num2word(p[0, 0])
                print(q_string)
                if q:
                    print("Correct answer:", a_string)
                    print("Model answer:", model_answer, 'with certainty', c[0, 0])
            print("Validation loss", valid_loss_acc / valid_size)
            print('#' * 80)

        elif step % 10 == 9:
            l, _, summary = sess.run([loss, optimize, summaries], feed_dict=train_feed)
            train_writer.add_summary(summary, step)

        else:
            l, _ = sess.run([loss, optimize], feed_dict=train_feed)

    train_writer.close()
