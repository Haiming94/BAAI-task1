import os
import re
import csv
import numpy as py
import pickle
import jieba
import collections
from langconv import *
import numpy as np
import pandas as pd
from string import punctuation


def Traditionl2Simplified(sentence):
    '''
    Traditionl --> Simplified
    :param sentence:
    :return:
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def Simplified2Traditionl(sentence):
    '''
    Simplified --> Traditionl
    :param sentence:
    :return:
    '''
    sentence = Converter('zh-hant').convert(sentence)
    return sentence


class reader(object):
    def __init__(self, filePath):
        self.filePath = filePath
        self.loadData()
        self.saveData()

    def loadData(self):
        print('reading data ...')
        with open(self.filePath+'train.csv', 'r', encoding='utf-8') as file:
            fileCsv = csv.reader(file)
            self.dataList = []
            i = 1
            for line in fileCsv:
                i = i + 1
                try:
                    if line[0] != None:
                        if len(line[1]) > 5:
                            if line[2] == '0' or line[2] == '1':
                                data = line[0]+' |||| '+line[1]+' |||| '+line[2]+'\n'
                                self.dataList.append(data)
                except:
                    print('{}, {}'.format(i, line))
                    continue
        print('the number of simples : {} / {} all'.format(len(self.dataList), i))

    def saveData(self):
        print('saving data ...')
        save_train = open(self.filePath + 'train.txt', 'w', encoding='utf-8')
        save_test = open(self.filePath + 'test.txt', 'w', encoding='utf-8')
        for line_id in range(len(self.dataList)):
            if line_id < (len(self.dataList) - 5000):
                save_train.write(self.dataList[line_id])
            else:
                save_test.write(self.dataList[line_id])
        print('save data, OK!')


# 定义要删除的标点等字符
add_punc='，。、【 】 “”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥'
all_punc=punctuation+add_punc


def loadFlie(datasets, rmPunc=True, is_train=True):
    if is_train:
        dataText, dataLabel, dataId = [], [], []
        for data in datasets:   # train, test
            text, labels, Ids = [], [], []
            with open(data, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.replace('\n', '').split('||||')
                    if len(line) == 3:
                        word = ' '.join(jieba.cut(line[1].strip()))
                        words = word.split(' ')
                        # remove symbols
                        if rmPunc:
                            for w in words:
                                if w in all_punc:
                                    words.remove(w)
                        text += [words]
                        labels += [int(line[2].strip())]
                        Ids += [line[0].strip()]
            dataText += [text]
            dataLabel += [labels]
            dataId += [Ids]
            # print('tst == {} == {}'.format(text[:2], len(text)))
            # print('tst == {} == {}'.format(Ids[:2], len(Ids)))
            # print('tst == {} == {}'.format(labels[:2], len(labels)))
            del word, words, text, labels, Ids
        return dataText, dataLabel, dataId
    else:
        data = pd.read_csv(datasets)
        shape = data.shape
        print(data.shape)
        id = [str(data.loc[i, ['id']].values[0]) for i in range(shape[0])]
        x_txt = [str(data.loc[i, ['text']].values[0]) for i in range(shape[0])]
        text = []
        for sen in x_txt:
            word = ' '.join(jieba.cut(sen.strip()))
            word = word.split(' ')
            text += [word]
        # the number of samples
        n_samples = len(text)
        lab = [int(1) for i in range(len(text))]
        # print('tst == {} == {}'.format(text[:2], len(x_txt)))
        # print('tst == {} == {}'.format(id[:2], len(id)))
        # print('tst == {} == {}'.format(lab[:2], len(lab)))
        return text, id, lab
    """
    data = pd.read_csv('debunking.csv')
    shape = data.shape
    print(data.shape)
    id = [str(data.loc[i, ['id']].values[0]) for i in range(shape[0])]
    # x_txt = [data.loc[i, ['text']].values[0] for i in range(shape[0])]
    # x_txt[:2]
    """
    

def insertWord(datasets):
    # insert all words
    allWords = []
    for data in datasets:
        # print('len of the data is:{}'.format(len(data)))
        for lines in data:
            allWords += lines

    # obtain frequent words
    counter = collections.Counter(allWords)
    vocab = len(counter)
    common_word = dict(counter.most_common(vocab - 2))  # 生成对应的词语字典
    print('the number of common words: {}'.format(len(common_word)))
    # number them
    c = 2
    for key in common_word:
        common_word[key] = c
        c += 1
    print('c = {}'.format(c))
    return common_word


def words2Number(text, label, id, comWord):
    transfText, transfLabel, transfId = [], [], []
    for lines, labels, ids in zip(text, label, id):
        new_x, new_l, new_i = [], [], []
        for lin, lab, idi in zip(lines, labels, ids):
            words = [comWord[w] if w in comWord else 1 for w in lin]
            new_x += [words]
            new_l += [lab]
            new_i += [idi]

        transfText += [new_x]
        transfLabel += [new_l]
        transfId += [new_i]
    return transfText, transfLabel, transfId


def word2vec(path, comWord):
    emDict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            emDict[word] = embedding
    word2vecs = [np.random.normal(0, 0.1, 300).tolist(), np.random.normal(0, 0.1, 300).tolist()]
    missing = 0
    for id, word in sorted(zip(comWord.values(), comWord.keys())):
        try:
            word2vecs.append(emDict[word])
        except KeyError:
            word2vecs.append(np.random.normal(0, 0.1, 300).tolist())
            missing += 1
    pickle.dump(word2vecs, open('./task1/dataset_vectors', 'wb'))
    print('missing: {}'.format(missing))
    print(np.array(word2vecs).shape)


if __name__ == "__main__":
    # read = reader('./task1/')
    datasets = ['./task1/train.txt', './task1/test.txt']
    text, label, id = loadFlie(datasets, rmPunc=False)
    tst_text, tst_id, tst_label = loadFlie('./task1/test_stage1.csv', rmPunc=False, is_train=False)
    text += [tst_text]
    label += [tst_label]
    id += [tst_id]
    print('text {}, label {}, id {}'.format(len(text), len(label), len(id)))
    comWord = insertWord(text)

    # write out filtering training test data
    transfText, transfLabel, transfId = words2Number(text, label, id, comWord)
    pickle.dump((transfText, transfLabel, transfId), open('./task1/dataset', 'wb'))

    embeddingPath = '/data/rali7/Tmp/wuhaiming/sgns.sogou.word'
    word2vec(embeddingPath, comWord)