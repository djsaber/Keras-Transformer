# -*- coding: gbk -*-
from utils import *
from transformer import *
from keras.optimizers import Adam
from keras.models import load_model
import os
import json


if __name__ == "__main__": 

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    # ���ó�����
    vocab_size = 12000
    seq_len = 64
    model_dim = 256
    n_heads = 8
    n_blocks = 6
    inner_dim = 1024
    sample_num = 500000
    batch_size = 128
    zh_txt_list=['�����������˵���¡�', 
                 'ʲô�£�', 
                 '��ϲ���㣡', 
                 '���ǲ�����������ˣ�',
                 '������˵��',
                 '�ټ���']


    # ��������
    # en, zh = load_NMT()
    en, zh = load_VATEX() 
    # en, zh = load_AI_Challenger_Translation_2017(sample_num)
    sample_num = len(en)


    # ����/�����ֵ�
    # zh_dic = build_dic(zh, vocab_size)
    # en_dic = build_dic(en, vocab_size)
    zh_dic = json.load(open('dict/zh_dic.json', 'r'))
    en_dic = json.load(open('dict/en_dic.json', 'r'))


    # ���ģ��
    # model = load_model('model/model_100.h5', custom_objects={'Transformer':Transformer})
    model = build_model(vocab_size, seq_len, model_dim, n_blocks, n_heads, inner_dim)
    model.summary()


    # �����Ż���
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-9)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )


    # �����������ͻص���
    data_gen = data_generater([zh, en], [zh_dic, en_dic], seq_len, batch_size)
    test_zh2en = test_callback(model, zh_txt_list, seq_len, [zh_dic, en_dic])
    set_lr = set_lr_callback()


    # ��ʼѵ��ģ��
    model.fit(data_gen, 
              epochs = 100, 
              steps_per_epoch = sample_num // batch_size,
              callbacks = [test_zh2en, set_lr])