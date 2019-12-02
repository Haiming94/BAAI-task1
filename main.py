import os
import pickle
import numpy as np
import tensorflow as tf
from config import *
import time
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from data_helpers import *


class Classifer(object):
    def __init__(self, config, session):
        self.inputText = tf.placeholder(tf.int32, [None, None], name='text')
        self.label = tf.placeholder(tf.int64, [None,], name='label')
        self.length = tf.placeholder(tf.int32, [None,], name='len')
        self.dropout = tf.placeholder(tf.float32, name='keep_prob')
        self.config = config
        self.lr = tf.Variable(0.0, trainable=False)
        self.embeddings = tf.get_variable("embeddings", shape=[config.vocab_size, config.hidden_size], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, trainable=config.embedding_trainable)
        self.input_layer()
        self.group()

    def input_layer(self):
        # input layer
        initial_hidden_states = tf.nn.embedding_lookup(self.embeddings, self.inputText) #[B, H, L]
        initial_cell_states = tf.identity(initial_hidden_states)
        self.initial_hidden_states= tf.nn.dropout(initial_hidden_states, self.dropout)
        self.initial_cell_states = tf.nn.dropout(initial_cell_states, self.dropout)
    
    def group(self):
        if self.config.model == 'cnn':
            initial_hidden_states=tf.reshape(self.initial_hidden_states, [-1, 700, config.hidden_size])            
            initial_hidden_states = tf.expand_dims(initial_hidden_states, -1)
            pooled_outputs = []
            for i, filter_size in enumerate([3]):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, config.hidden_size, 1, config.hidden_size]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[config.hidden_size]), name="b")

                    W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
                    b2 = tf.Variable(tf.constant(0.1, shape=[config.hidden_size]), name="b2")

                    W3 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W3")
                    b3 = tf.Variable(tf.constant(0.1, shape=[config.hidden_size]), name="b3")

                    W4 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W4")
                    b4 = tf.Variable(tf.constant(0.1, shape=[config.hidden_size]), name="b4")

                    conv = tf.nn.conv2d(
                        initial_hidden_states,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    print(h.get_shape())
                    h=tf.transpose(h, [0,1,3,2])
                    # Apply nonlinearity


                    conv2 = tf.nn.conv2d(
                        h,
                        W2,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv2")
                    h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
                    print(h2.get_shape())
                    h2=tf.transpose(h2, [0,1,3,2])

                    conv3 = tf.nn.conv2d(
                        h2,
                        W3,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv3")  
                    h3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu3")
                    print(h3.get_shape())

                    # Max-pooling over the outputs
                    pooled = tf.nn.max_pool(
                        h3,
                        ksize=[1, 700 - 3*filter_size + 3, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = 1 * config.hidden_size
            self.h_pool = tf.concat(pooled_outputs, axis=3)
            representation = tf.reshape(self.h_pool, [-1, num_filters_total])

            softmax_w = tf.Variable(tf.random_normal([2*config.hidden_size, config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_w")
            softmax_b = tf.Variable(tf.random_normal([config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_b")

            softmax_w2 = tf.Variable(tf.random_normal([config.hidden_size, 2*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_w2")
            softmax_b2 = tf.Variable(tf.random_normal([2*config.hidden_size], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_b2")
            representation=tf.nn.tanh(tf.matmul(representation, softmax_w2)+softmax_b2)

        elif self.config.model == 'lstm':
            initial_hidden_states=self.lstm_layer(self.initial_hidden_states, self.config, self.dropout, self.length)
            softmax_w = tf.Variable(tf.random_normal([2*config.hidden_size, config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_w")
            softmax_b = tf.Variable(tf.random_normal([config.num_label], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32, name="softmax_b")
            representation=tf.reduce_sum(initial_hidden_states,axis=1)
            config.hidden_size_sum=2*config.hidden_size

        else:
            print('Invalid model')
            exit(1)

        logits = tf.matmul(representation, softmax_w) + softmax_b
        self.to_print=tf.nn.softmax(logits)
        #operators for prediction
        self.prediction=prediction=tf.argmax(logits,1)
        correct_prediction = tf.equal(prediction, self.label)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))      # 准确率
        
        #cross entropy loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=logits)
        self.cost=cost=tf.reduce_mean(loss)+ config.l2_beta*tf.nn.l2_loss(self.embeddings)    # 损失的均值+词向量的l2范数

        #designate training variables
        tvars=tf.trainable_variables()      # tf.trainable_variables返回所有 当前计算图中 在获取变量时未标记 trainable=False 的变量集合
        self.lr = tf.Variable(0.0, trainable=False)     # lr不被训练
        grads=tf.gradients(cost, tvars)     # 求cost 关于tvars 对应变量的导数
        grads, _ = tf.clip_by_global_norm(grads, config.max_grad_norm)       # 为了处理gradient explosion或者gradients vanishing的问题
        self.grads=grads
        optimizer = tf.train.AdamOptimizer(config.learning_rate)        
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))        # 返回(gradient, variable) 应用指定梯度的操作


    def lstm_layer(self, x, config, keep_prob, length):
        with tf.variable_scope('forward'):
            fw_lstm = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0)
            # fw_lstm = LSTMCell(config.hidden_size, Weight_fw, forget_bias=0.0)
            fw_lstm = tf.contrib.rnn.DropoutWrapper(fw_lstm, output_keep_prob=keep_prob)

        with tf.variable_scope('backward'):
            bw_lstm = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0)
            # bw_lstm = LSTMCell(config.hidden_size, Weight_bw, forget_bias=0.0)
            bw_lstm = tf.contrib.rnn.DropoutWrapper(bw_lstm, output_keep_prob=keep_prob)
        
        #bidirectional rnn
        with tf.variable_scope('bilstm'):
            lstm_output=tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, inputs=x, sequence_length=length, time_major=False, dtype=tf.float32)
            lstm_output=tf.concat(lstm_output[0], 2)

        return lstm_output

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

def get_minibatches_idx(n, config, shuffle):
    idx_list = np.arange(n, dtype='int32')

    if shuffle:
        np.random.shuffle(idx_list)
    
    minibatches = []
    minibatch_start = 0
    n_batch = n // config.batch_size
    for i in range(n_batch):
        minibatches += [idx_list[minibatch_start: minibatch_start+config.batch_size]]
        minibatch_start += config.batch_size
    if (minibatch_start != n):
        n_batch += 1
        minibatches += [idx_list[minibatch_start:]]
    return minibatches, n_batch


def run_epoch(session, config, model, data, train_op, is_training):
    n_samples = len(data[0])
    print('Running {} samples:'.format(n_samples))
    minibatches, n_batch = get_minibatches_idx(n_samples, config, shuffle=False)

    if is_training == False:
        config.keep_prob = 1
        print('testing...')

    correct = 0.
    total = 0
    total_cost = 0
    prediction = []
    for batch_id in tqdm(range(n_batch)):
        inds = minibatches[batch_id]
        x = data[0][inds]
        if config.model == 'cnn':
            x = pad_sequences(x, maxlen=700, dtype='int32', padding='post', truncating='post', value=0.)
        else:
            x = pad_sequences(x, maxlen=None, dtype='int32', padding='post', truncating='post', value=0.)
        y = data[1][inds]
        length = data[2][inds]

        count, _, cost, to_print= \
        session.run([model.accuracy, train_op, model.cost, model.to_print],\
            {model.inputText: x, model.label: y, model.length:length, model.dropout:config.keep_prob})        	        # eval_op 梯度信息
        # to_print = session.run(to_print)

        correct += count
        total += len(inds)
        total_cost += cost
        prediction += to_print.tolist()
    accuracy = correct / total
    
    return accuracy, prediction

def train_test_model(config, i, session, model, trainData, devData):
    # compute lr_decay
    lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)

    # training
    print('\nEpoch: {} Learning rate: {:.4f}'.format(i+1, session.run(model.lr)))
    start_time = time.time()
    train_acc, _ = run_epoch(session, config, model, trainData, model.train_op, True)
    print('train accuracy: {:.2f}%, time: {:.3f} seconds\n'.format(train_acc*100, time.time()-start_time))

    # dev
    dev_acc, _ = run_epoch(session, config, model, devData, tf.no_op(), False)
    print('dev accuracy: {:.2f}% \n'.format(dev_acc*100))

    return dev_acc
    


def start_epoches(config, session, classifier, trainData, devData, testData):
    saver = tf.train.Saver()
    acc_save = 0.
    for i in range(config.max_max_epoch):
        acc = train_test_model(config, i, session, classifier, trainData, devData)
        if acc > acc_save and i > 1:  #  
        # if acc > acc_save:  #
            _, prediction = run_epoch(session, config, classifier, testData, tf.no_op(), False)
            print('test {} is ok!\n'.format(i+1))
            save_path = open('./log/'+str(i+1)+'_pre09.txt', 'w')
            if len(prediction) == len(tstId):
                print('all right!')
            else:
                print('ids -- {}  pre -- {}'.format(len(tstId), len(prediction)))
            for idx in range(len(prediction)):
                pre_test = str(tstId[idx]) + ', ' + str(prediction[idx][0]) +', '+ str(prediction[idx][1]) + '\n'
                print(pre_test)
                save_path.write(pre_test)
                save_path.flush()
            # save model parameters
            acc_save = acc
            saver.save(session, './log/'+config.model+'_09.ckpt')


def word_to_vec(embedMatrix, session, config, *args):
    print('word2vec shape: {}'.format(embedMatrix.shape))
    for model in args:
        session.run(tf.assign(model.embeddings, embedMatrix))

if __name__ == "__main__":
    # configs
    config = Config()

    # load embeddings
    vectorPath = config.data_path + 'dataset_vectors'
    file = open(vectorPath, 'rb')
    embedMatrix = np.array(pickle.load(file))
    config.vocab_size = embedMatrix.shape[0]

    # load train dev dataset
    dataPath = config.data_path + 'dataset'
    trainData, devData = loadData(dataPath, config.vocab_size)
    trainData = prepareData(trainData[0], trainData[1])
    devData = prepareData(devData[0], devData[1])

    # load dataset
    testData = loadData(dataPath, config.vocab_size, is_train=False)
    tstId = testData[2]
    save_id_file = open('./log/id_list.txt', 'w')
    for id in tstId:
        save_id_file.write(id+'\n')
        save_id_file.flush()
    save_id_file.close()
    print('id -- {}, label -- {}, text -- {}'.format(len(tstId), len(testData[1]), len(testData[0])))
    testData = prepareData(testData[0], testData[1])

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    gpu_options = tf.GPUOptions(allow_growth=True)  
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        initializer = tf.random_normal_initializer(0, 0.05)

        classifier = Classifer(config=config, session=session)

        total = 0
        # print trainable variables
        for v in tf.trainable_variables():
            print(v.name)
            shape = v.get_shape()
            try:
                size = shape[0].value * shape[1].value
            except:
                size = shape[0].value
            total += size
        print(total)

        # initialize
        init = tf.global_variables_initializer()
        session.run(init)
        # train test model
        word_to_vec(embedMatrix, session, config, classifier)
        start_epoches(config, session, classifier, trainData, devData, testData)
        # start_epoches(config, session, classifier, devData, devData, testData)
