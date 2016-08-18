import tensorflow as tf
import numpy as np

from babi_batch_generator import bAbiBatchGenerator, num2word
from basic_layers import LSTM, SparseGRU, FeedForward, ConnectLayers

batch_size = 32

graph = tf.Graph()
gen = bAbiBatchGenerator(1, batch_size)
eval_gen = bAbiBatchGenerator(1, 1, 'test')

sentence_encoding_size = 200
vocabulary_size = gen.vocabulary_size
sentence_padding = gen.padding
num_unrollings = 15


def conditional_sparse_softmax_ce_multiple_choise(logits, labels, n_labels, weights, cond):
    def compute_loss():
        labels_list = list(map(lambda l: tf.squeeze(l, [1]), tf.split(1, n_labels, labels)))
        loss = tf.zeros([1])
        for i in range(n_labels):
            loss += weights[i] * tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_list[i]))
        return loss

    return tf.cond(cond, compute_loss, lambda: tf.zeros([1]))

with graph.as_default():
    encoder = SparseGRU(vocabulary_size, sentence_encoding_size, batch_size)
    layer1 = LSTM(sentence_encoding_size, 200, batch_size)
    layer2 = FeedForward(200, vocabulary_size, batch_size)
    model = ConnectLayers((layer1, layer2))

    input_facts = []
    labels = []
    question_marks = []
    new_story_marks = []

    losses = []
    predictions = []
    for i in range(num_unrollings):
        fact = tf.placeholder(tf.int32, (batch_size, sentence_padding))
        input_facts.append(fact)
        is_new_story = tf.placeholder(tf.bool, [])
        new_story_marks.append(is_new_story)

        encoding = encoder.feed_sequence_tensor(fact, sentence_padding)[-1]
        encoder.reset_current_state()

        model.reset_current_state_if(is_new_story)
        output = model.feed_input(encoding)

        label = tf.placeholder(tf.int32, (batch_size, 2))
        labels.append(label)
        is_question = tf.placeholder(tf.bool, [])
        question_marks.append(is_question)

        losses.append(conditional_sparse_softmax_ce_multiple_choise(output, label, 2, [1, 0], is_question))

    loss = tf.reduce_mean(tf.concat(0, losses))  # + regularizer

    with tf.control_dependencies([model.save_state()]):
        optimize = tf.train.AdadeltaOptimizer(0.01).minimize(loss)

    valid_input = tf.placeholder(tf.int32, (1, sentence_padding))
    valid_encoding = encoder.eval_model.feed_sequence_tensor(valid_input, sentence_padding)[-1]
    encoder.eval_model.reset_current_state()

    valid_output = model.eval_model.feed_input(valid_encoding)
    valid_label = tf.placeholder(tf.int32, (1, 2))
    valid_is_question = tf.placeholder(tf.bool, [])

    with tf.control_dependencies([model.eval_model.save_state()]):
        valid_loss = conditional_sparse_softmax_ce_multiple_choise(
            valid_output, valid_label, 2, [1, 0], valid_is_question)

        valid_probs = tf.nn.softmax(valid_output)
        prediction_certainty, valid_prediction = tf.nn.top_k(valid_probs, 2)


num_steps = 20000
summary_frequency = 500
valid_size = 15

with tf.Session(graph=graph) as sess:
    tf.initialize_all_variables().run()

    for step in range(num_steps):
        feed = dict()
        for j in range(num_unrollings):
            sentences, answers, is_question, is_new_story = gen.get_next_batch()
            feed[input_facts[j]] = sentences
            feed[labels[j]] = answers
            feed[question_marks[j]] = bool(is_question)
            feed[new_story_marks[j]] = bool(is_new_story)

        l, _ = sess.run([loss, optimize], feed_dict=feed)
        if step % summary_frequency == 0:
            print(step, l)
            valid_loss_acc = 0.
            for i in range(valid_size):
                f, a, q, s = eval_gen.get_next_batch()
                q_string, a_string = eval_gen.batch2strings((f, a, q, s))[0]
                feed = dict()
                feed[valid_input] = f
                feed[valid_label] = a
                feed[valid_is_question] = q
                l, c, p = sess.run([valid_loss, prediction_certainty, valid_prediction], feed_dict=feed)
                valid_loss_acc += l
                model_answer = num2word(p[0, 0])
                print(q_string)
                if q:
                    print("Correct answer:", a_string)
                    print("Model answer:", model_answer, 'with certainty', c[0, 0])
            print("Validation loss", valid_loss_acc / valid_size)
            print('#' * 80)

