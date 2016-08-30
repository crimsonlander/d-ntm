import numpy as np
import os
import re

max_fact_len = [7, 7, 8, 8, 8, 7, 7, 7, 8, 10, 9, 9, 9, 9, 6, 5, 12, 11, 11, 8]
max_story_len = [15, 73, 325, 3, 131, 31, 57, 55, 15, 15, 15, 15, 15, 19, 12, 10, 10, 15, 6, 24]

dir_name = 'tasks_1-20_v1-2/en-10k'
all_file_names = os.listdir(dir_name)
train_suffix = 'train.txt'
test_suffix = 'test.txt'


def question_number(filename):
    i = j = 2
    while filename[j].isnumeric():
        j += 1
    return int(filename[i:j])

train_file_names = sorted(
    filter(lambda n: n[-len(train_suffix):] == train_suffix, all_file_names),
    key=question_number)

test_file_names = sorted(
    filter(lambda n: n[-len(test_suffix):] == test_suffix, all_file_names),
    key=question_number)

words = set()
files = []
for name in all_file_names:
    files.append(open(dir_name + '/' + name).read().lower())
    file_words = re.split('\W+', files[-1])
    words |= set(filter(lambda w: not w.isnumeric(), file_words))

words.remove('')
words.add('.')
words.add('?')
_num2word = dict(enumerate(sorted(words), 1))
_num2word[0] = ''


def num2word(x):
    return _num2word[x]

_word2num = dict(zip(_num2word.values(), _num2word.keys()))
_num2word[0] = '_'

def word2num(x):
    return _word2num[x]

vocabulary_size = len(_num2word)


def parse_file(filename):
    stories = []
    file = open(dir_name + '/' + filename).read().lower()
    qa = file.split(sep='\t')
    q = qa[:-1:2]
    a = qa[1::2]
    for i in range(len(q)):
        answer = list(a[i].split(','))
        facts = list(map(lambda x: x.split(), q[i].split('\n')))
        if i > 0:
            facts = facts[1:]

        if facts[0][0] == '1':
            stories.append([])

        def clean_fact(fact):
            fact = fact[1:]
            sign = fact[-1][-1]
            fact[-1] = fact[-1][:-1]
            fact.append(sign)
            return [word2num(w) for w in fact]

        facts = list(map(clean_fact, facts))
        answers = [[] for _ in range(len(facts))]
        answers[-1] = [word2num(w) for w in answer]

        stories[-1] += list(zip(facts, answers))

    return stories


def story2array(story, fact_padding):
    facts = []
    answers = []
    alignment = [0]

    for i in range(len(story)):
        fact = np.zeros((1, fact_padding), dtype=np.int32)
        fact_len = len(story[i][0])
        fact[0, -fact_len:] = np.asarray(story[i][0], dtype=np.int32)
        answer = np.zeros((1, 2), dtype=np.int32)
        if len(story[i][1]):
            alignment.append(i)
        for j in range(len(story[i][1])):
            answer[0, j] = story[i][1][j]
        facts.append(fact)
        answers.append(answer)

    alignment.append(len(story))
    return facts, answers, alignment


def align_arrays(arrays, alignments, padding=0):
    n_questions = max(map(len, alignments)) - 1

    spacings = []

    for i in range(len(alignments)):
        spacing = []
        for j in range(1, len(alignments[i])):
            l = alignments[i][j-1]
            r = alignments[i][j]
            spacing.append(r - l)
        for _ in range(len(alignments[i]) - 1, n_questions):
            spacing.append(0)
        spacings.append(spacing)

    global_alignment = [0]
    global_spacing = []
    for i in range(n_questions):
        max_spacing = max([spacing[i] for spacing in spacings])
        global_spacing.append(max_spacing)
        global_alignment.append(global_alignment[-1] + max_spacing)

    result = []

    for i in range(n_questions):
        columns = [[] for _ in range(global_spacing[i])]
        for j in range(len(alignments)):
            l = min(i, len(alignments[j]) - 1)
            r = min(i + 1, len(alignments[j]) - 1)

            row = arrays[j][alignments[j][l]:alignments[j][r]]
            row_padding = []
            for _ in range(global_spacing[i] - len(row)):
                row_padding.append(np.zeros((1, padding), dtype=np.int32))
            row = row_padding + row

            for k in range(global_spacing[i]):
                columns[k].append(row[k])

        for k in range(global_spacing[i]):
            result.append(np.concatenate(columns[k]))

    question_marks = [0] * len(result)
    for i in global_alignment[1:-1]:
        question_marks[i] = 1

    return result, question_marks


class bAbiBatchGenerator:
    def __init__(self, task_number, batch_size, file_type='train'):
        assert 1 <= task_number <= 20
        task_number -= 1
        if file_type == 'train':
            file_names = train_file_names
        else:
            assert file_type == 'test'
            file_names = test_file_names

        self.padding = max_fact_len[task_number]
        self.batch_size = batch_size
        self.stories = parse_file(file_names[task_number])
        self.stories_index = 0
        self.facts = []
        self.answers = []
        self.question_marks = []
        self.content_length = 0
        self.content_index = 0
        self.vocabulary_size = vocabulary_size

    def load_new_content(self):
        self.content_index = 0

        facts = []
        answers = []
        alignments = []
        for i in range(self.batch_size):
            story = self.stories[self.stories_index]
            self.stories_index = (self.stories_index + 1) % len(self.stories)
            f, an, al = story2array(story, self.padding)

            facts.append(f)
            answers.append(an)
            alignments.append(al)

        self.facts, self.question_marks = align_arrays(facts, alignments, self.padding)
        self.answers, _ = align_arrays(answers, alignments, 2)
        self.content_length = len(self.facts)

    def get_next_batch(self):
        new_story = 0
        if self.content_index == self.content_length:
            self.load_new_content()
            new_story = 1
        batch = (self.facts[self.content_index],
                 self.answers[self.content_index],
                 self.question_marks[self.content_index],
                 new_story)
        self.content_index += 1
        return batch

    @staticmethod
    def batch2strings(batch):
        sentence, answer, is_a_question, is_new_story = batch

        strings = []
        for i in range(len(sentence)):
            sentence_str = ' '.join(map(num2word, sentence[i, :]))
            answer_str = ''
            if is_a_question:
                answer_str = ' ' + ' '.join(map(num2word, answer[i, :]))
            strings.append((sentence_str.strip(), answer_str.strip()))

        return strings
