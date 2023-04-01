# -*- coding: gbk -*-
import spacy
import zhconv
import json
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from collections import Counter
from keras.callbacks import Callback, LearningRateScheduler
from keras.losses import SparseCategoricalCrossentropy


def load_AI_Challenger_Translation_2017(sample_num=200000):
    '''��ȡAI Challenger Translation 2017���ݼ������ı����зִʣ����б���
    ��������

    ���أ�
        - en_list = [['I', 'have', 'a', 'dog', '.'], [...]]
        - zh_list = [['��', '��', 'һֻ', 'С��', '��'], [...]]     
    '''

    path = 'D:/����/python����/�����ֲ�/���ݼ�/AI Challenger Translation 2017/en_to_zh/dataset/'
    spacy_en = spacy.load("en_core_web_sm")
    spacy_zh = spacy.load("zh_core_web_sm")

    with open(path+'train.en', 'r', encoding='utf-8') as f:
        en_corpus = f.readlines()
    with open(path+'train.zh', 'r', encoding='utf-8') as f:
        zh_corpus = f.readlines()
    assert len(en_corpus) == len(zh_corpus)

    en_zh_corpus = list(zip(en_corpus, zh_corpus))
    random.shuffle(en_zh_corpus)
    en_corpus, zh_corpus = zip(*en_zh_corpus)
    en_corpus, zh_corpus = en_corpus[:sample_num], zh_corpus[:sample_num]
    print(f'ʾ����{en_corpus[0]}\t{zh_corpus[0]}')
    print(f'ʾ����{en_corpus[-1]}\t{zh_corpus[-1]}')
    en_list, zh_list = [], []
    for en_sent, zh_sent in tqdm(zip(en_corpus, zh_corpus), desc='loading', total=len(en_corpus)):
        en_sent = [tok.text for tok in spacy_en.tokenizer(en_sent[:-1])]
        zh_sent = [tok.text for tok in spacy_zh.tokenizer(zh_sent[:-1])]
        en_list.append(en_sent)
        zh_list.append(zh_sent)
    print(f'ʾ����{en_list[0]}\t{zh_list[0]}')
    print(f'ʾ����{en_list[-1]}\t{zh_list[-1]}')
    print('�������ݼ�[AI Challenger Translation 2017]��ɣ�')
    return en_list, zh_list


def load_VATEX():
    '''��ȡVATEX���ݼ������ı����зִʣ����б���
    ��������

    ���أ�
        - en_list = [['I', 'have', 'a', 'dog', '.'], [...]]
        - zh_list = [['��', '��', 'һֻ', 'С��', '��'], [...]]       
    '''

    path = 'D:/����/python����/�����ֲ�/���ݼ�/VATEX/data/'
    spacy_en = spacy.load("en_core_web_sm")
    spacy_zh = spacy.load("zh_core_web_sm")
    with open(path + 'vatex_training_v1.0.json', 'r', encoding = 'utf-8') as f:
        en_list, zh_list = [], []
        data = json.load(f)
        for move in tqdm(data, desc='loading VATEX'):
            for en_sent, zh_sent in zip(move["enCap"], move["chCap"]):
                en_sent = [tok.text for tok in spacy_en.tokenizer(en_sent)]
                zh_sent = [tok.text for tok in spacy_zh.tokenizer(zh_sent)]
                en_list.append([w for w in en_sent if w != ' '])
                zh_list.append(zh_sent)
        print(f'ʾ����{en_list[0]}\t{zh_list[0]}')
        print(f'ʾ����{en_list[-1]}\t{zh_list[-1]}')
        print('�������ݼ�[VATEX]��ɣ�')
        return en_list, zh_list


def load_NMT():
    '''��ȡNMT���ݼ������ı����зִʣ����б���
    ��������

    ���أ�
        - en_list = [['I', 'have', 'a', 'dog', '.'], [...]], 
        - zh_list = [['��', '��', 'һֻ', 'С��', '��'], [...]], 
    '''

    path = "D:/����/python����/�����ֲ�/���ݼ�/NMT/en-cn/"
    spacy_en = spacy.load("en_core_web_sm")
    spacy_zh = spacy.load("zh_core_web_sm")
    with open(path + 'cmn.txt', "r", encoding="utf-8") as f:
        en_list, zh_list = [], []
        for line in tqdm(f.readlines(), desc='loading'):
            en_sent, zh_sent = line[:-1].split('\t')
            zh_sent = zhconv.convert(zh_sent, 'zh-cn')  
            en_sent = [tok.text for tok in spacy_en.tokenizer(en_sent)]
            zh_sent = [tok.text for tok in spacy_zh.tokenizer(zh_sent)]
            en_list.append([w for w in en_sent if w != ' '])
            zh_list.append(zh_sent)
        print(f'ʾ����{en_list[0]}\t{zh_list[0]}')
        print(f'ʾ����{en_list[-1]}\t{zh_list[-1]}')
        print(f'�������ݼ�[NMT]��ɣ�')
        return en_list, zh_list


def build_dic(sent_list, vocab_size, S='BOS', E='EOS', P='PAD', U='UNK', case_sensitive=False):
    '''������Ƶ�Σ������ʱ�key�����ʣ�value��id
    ������
        - sent_list���ִʺ�ľ����б����磺[['I', 'love', 'you', '.'], [...], ...], 
        - vocab_size���ֵ��С�����ɵĵ�������
        - S��'BOS' ��ʼ��־ 
        - E��'EOS' ������־ 
        - P��'PAD' ����־ 
        - U��'UNK' δ֪��־
        - case_sensitive��False�����ִ�Сд�����磺'the' �� 'The'

    ���أ�vocab_dict�����ʱ�       
    '''

    dic = {'A':'a', 'B':'b', 'C':'c', 'D':'d', 'E':'e', 'F':'f', 'G':'g', 'H':'h', 'I':'i', 'J':'j', 'K':'k', 
           'L':'l', 'M':'m', 'N':'n', 'O':'o', 'P':'p', 'Q':'q', 'R':'r', 'S':'s', 'T':'t', 'U':'u', 'V':'v', 
           'W':'w', 'X':'x', 'Y':'y', 'Z':'z', }
    _dic = {item[1]:item[0] for item in dic.items()}

    if not case_sensitive:
        sent_list = [[w if w[0] not in dic.keys() else dic[w[0]]+w[1:] for w in sent] for sent in sent_list]

    vocab_dict = {P:0, S:1, E:2, U:3, }
    word_count = Counter([word for sent in sent_list for word in sent])
    ls = word_count.most_common(int(vocab_size-4))
    print(f'���ϵ���������{len(word_count)}  ��ʹ�Ƶ��{ls[-1][-1]}  �ֵ��С��{vocab_size}')

    for i, w in enumerate(ls):
        w = w[0]
        vocab_dict[w] = i + 4
        if not case_sensitive:
            w = _dic[w[0]]+w[1:] if w[0] in _dic.keys() else w
            vocab_dict[w] = i + 4
    # vocab_dict.update({w[0]: i + 4 for i, w in enumerate(ls)})
    return vocab_dict


def sent2vec(sent_list, vec_len, vocab_dict, S='BOS', E='EOS', P='PAD', U='UNK'):
    '''�����Ӱ����ֵ�ӳ��Ϊ����
    ������
        - sent_list: �ִʺ�ľ����б����磺[['I', 'love', 'you', '.'], [...], ...], 
        - len���������ȣ����ӳ����ó�����ضϣ���������
        - vocab_dict���ʱ�
        - S��'BOS' ��ʼ��־��Noneʱ���ʾ�������ʼ��־
        - E��'EOS' ������־��Noneʱ���ʾ����ӽ�����־
        - P��'PAD' ����־ 
        - U��'UNK' δ֪��־
    ���أ�
        - vec_list��������֮��ľ����б����磺[['1', '83', '72', '64', 2, 0, 0, ...], [...], ...], 
    '''

    # ���'BOS''EOS'��ʼ��־
    if S: sent_list = [[S]+sent for sent in sent_list]
    if E: sent_list = [sent+[E] for sent in sent_list]
    # ���'PAD'����־/�ضϳ�����
    sent_list = [sent+[P]*(vec_len-len(sent)) if len(sent)<vec_len else sent[:vec_len] for sent in sent_list]
    # ����vec_list
    vec_list = [[vocab_dict[w] if w in vocab_dict.keys() else vocab_dict[U] for w in sent] for sent in sent_list]
    return vec_list


def data_generater(data,
                   vocab_dics,
                   vec_len = 50,
                   batch_size = 32,
                   shuffle=True,
                   S = 'BOS',
                   E = 'EOS',
                   P = 'PAD',
                   U = 'UNK'):
    '''����ѵ������
    ���طִʺ���ı��б�����ѵ�����ݡ�����transformer�ı�ǩ��
    one-hot��ʾ������ռ�úܴ����Բ���data generater��ʽ��
    ������
         - data��[�����б�, Ӣ���б�]���ִʺ�
         - vocab_dics��[���Ĵʱ�, Ӣ�Ĵʱ�]
         - vec_len������������
         - batch_size������������ÿ�����ɵ�������
         - shuffle=True���������ݼ�
         - S = 'BOS' ��ʼ��־���ʹʱ�һ��
         - E = 'EOS' ������־���ʹʱ�һ��
         - P = 'PAD' ����־���ʹʱ�һ��
         - U = 'UNK' δ֪��־���ʹʱ�һ��
    '''

    zh, en = data
    assert len(zh) == len(en)
    zh_dic, en_dic = vocab_dics

    while True:
        if shuffle: sample_list = random.sample(range(len(en)), batch_size)
        else: sample_list = [i for i in range(batch_size)]
        sample_en, sample_zh = [en[i] for i in sample_list], [zh[i] for i in sample_list]
        zh_vec = sent2vec(sample_zh, vec_len, zh_dic, S, E, P, U)
        en_vec = sent2vec(sample_en, vec_len, en_dic, S, E, P, U)
        out_vec = sent2vec(sample_en, vec_len, en_dic, None, E, P, U)
        zh_vec = np.asarray(zh_vec, dtype='int16')
        en_vec = np.asarray(en_vec, dtype='int16')
        out_vec = np.asarray(out_vec, dtype='int16')
        yield [zh_vec, en_vec], out_vec
    

def test(zh_txt_list, max_len, vocab_dicts, model):
    '''zh-en���ķ����Ӣ�Ĳ���
    ������
        - zh_txt_list�������ı��б�, ['����һֻС����', '���ǳ��ɰ���', ...]
        - max_len��ģ��������������
        - [zh_vocab_dict, en_vocab_dict],��Ӣ�ĵ��ʱ�
        - model��transformerģ��
    '''
    spacy_zh = spacy.load("zh_core_web_sm")

    zh_dict, en_dict = vocab_dicts
    zh_list = [[tok.text for tok in spacy_zh.tokenizer(sent)] for sent in zh_txt_list]

    en_list = [[] for _ in range(len(zh_list))]
    zh_vec = sent2vec(zh_list, max_len, zh_dict)
    en_vec = sent2vec(en_list, max_len, en_dict, E=None)
    zh_vec = np.asarray(zh_vec, dtype='int16')
    en_vec = np.asarray(en_vec, dtype='int16')

    for i in range(max_len-1):
        en_out = model.predict([zh_vec, en_vec], verbose=0)
        w_id = np.argmax(en_out[:,i], axis=1)
        en_vec[:,i+1] = w_id

    _en_dict = {item[1]:item[0] for item in en_dict.items()}
    en_txt_list = [[_en_dict[wid] if wid in _en_dict.keys() else en_dict['UNK'] for wid in np.argmax(sent, axis=1)] for sent in en_out]
    en_txt_list = [[s for s in sent if s not in ['EOS', 'PAD']] for sent in en_txt_list]
    print(f'\ntrying to translate...')
    for zh, en in zip(zh_txt_list, en_txt_list):
        print(f'{zh}\t--->\t{en}')


class test_callback(Callback):
    """�ص���������ģ��
    ������
        - model��ѵ����ģ��
        - zh_txt_list�����������ı�
        - max_len����������󳤶�
        - vocab_dicts����Ӣ�Ĵʱ�
    """
    def __init__(self, model, zh_txt_list, max_len, vocab_dicts):
        self.model = model
        self.zh_txt_list = zh_txt_list
        self.max_len = max_len
        self.vocab_dicts = vocab_dicts
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % 5 == 0:
            test(self.zh_txt_list,
                 self.max_len,
                 self.vocab_dicts,
                 self.model)

        if (epoch+1) % 10 == 0:
            self.model.save(f'model/model_{epoch+1}.h5')


def set_lr_callback():
    def myScheduler(epoch):
        '''�Զ���ѧϰ��
        ʵ�ֶ�ѧϰ�ʵ�warm_up����
        '''
        ini_lr = 0.0001
        max_lr = 0.0001
        min_lr = 0.00005
        up_steps = 30
        down_steps = 50

        if epoch <= up_steps:
            lr = ini_lr + epoch * ((max_lr - ini_lr) / up_steps)
        elif (up_steps+down_steps) >= epoch > up_steps:
            lr = max_lr - (epoch - up_steps) * ((max_lr - min_lr) / down_steps)
        else:
            lr = min_lr

        print(f'learning rate:[{lr}]')
        return lr

    return LearningRateScheduler(myScheduler)


def loss_function(real, pred):
    '''�Զ�����ʧ����
    ����Ŀ����������䣨padded�����ģ�
    ����ڼ�����ʧ����ʱ��Ӧ������ڵ��ǳ���Ҫ��
    ''' 
    def smooth_labels(labels, factor=0.1):
        # smooth the labels
        labels = tf.cast(labels, tf.float32)
        labels *= (1 - factor)
        labels += (factor / tf.cast(tf.shape(labels)[-1], tf.float32))
        return labels

    mask = tf.math.logical_not(tf.math.equal(real, 0))  

    real = smooth_labels(real)
    loss = SparseCategoricalCrossentropy(from_logits=True, reduction='none')(real, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    # ************************************************************** #
    # mask = tf.cast(tf.not_equal(loss, 0), dtype=loss.dtype)
    # mask = tf.reduce_sum(mask, axis=1)
    # loss = tf.reduce_sum(loss, axis=1)
    # loss = tf.divide(loss, mask)
    # ************************************************************** #

    loss = tf.reduce_mean(loss)
    
    return loss