class Config(object):
    model = 'lstm'
    
    vocab_size = 15000
    hidden_size = 300
    max_epoch = 2
    max_max_epoch = 10
    batch_size = 32
    num_label = 2
    keep_prob = 0.5
    learning_rate = 0.001
    l2_beta = 0.0
    max_grad_norm = 5
    lr_decay = 0.95
    # embedding parameters 
    embedding_trainable = True


    data_path = './task1/'