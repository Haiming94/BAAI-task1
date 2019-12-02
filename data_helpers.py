import pickle
import numpy as np

def removeUnk(x, n_words):
    return [[1 if w >= n_words else w for w in sen] for sen in x]

def loadData(path, n, is_train=True):
    with open(path, 'rb') as file:
        dataText, dataLabel, dataId = pickle.load(file)
        if is_train:
            trnText, trnLabel, trnID = dataText[0], dataLabel[0], dataId[0]
            devText, devLabel, tstID = dataText[1], dataLabel[1], dataId[1]
            trnText = removeUnk(trnText, n)
            devText = removeUnk(devText, n)
            return [trnText, trnLabel, trnID], [devText, devLabel, tstID]
        else:
            tstText, tstLabel, tstID = dataText[2], dataLabel[2], dataId[2]
            tstText = removeUnk(tstText, n)
            print('load --> id -- {}, label -- {}, text -- {}'.format(len(tstID), len(tstLabel), len(tstText)))
            return [tstText, tstLabel, tstID]


def prepareData(text, label):
    length = [len(s) for s in text]
    labels = np.array(label).astype('int32')
    return [np.array(text), labels, np.array(length)]