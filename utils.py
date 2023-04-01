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
    '''读取AI Challenger Translation 2017数据集，对文本进行分词，以列表返回
    参数：无

    返回：
        - en_list = [['I', 'have', 'a', 'dog', '.'], [...]]
        - zh_list = [['我', '有', '一只', '小狗', '。'], [...]]     
    '''

    path = 'D:/科研/python代码/炼丹手册/数据集/AI Challenger Translation 2017/en_to_zh/dataset/'
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
    print(f'示例：{en_corpus[0]}\t{zh_corpus[0]}')
    print(f'示例：{en_corpus[-1]}\t{zh_corpus[-1]}')
    en_list, zh_list = [], []
    for en_sent, zh_sent in tqdm(zip(en_corpus, zh_corpus), desc='loading', total=len(en_corpus)):
        en_sent = [tok.text for tok in spacy_en.tokenizer(en_sent[:-1])]
        zh_sent = [tok.text for tok in spacy_zh.tokenizer(zh_sent[:-1])]
        en_list.append(en_sent)
        zh_list.append(zh_sent)
    print(f'示例：{en_list[0]}\t{zh_list[0]}')
    print(f'示例：{en_list[-1]}\t{zh_list[-1]}')
    print('加载数据集[AI Challenger Translation 2017]完成！')
    return en_list, zh_list


def load_VATEX():
    '''读取VATEX数据集，对文本进行分词，以列表返回
    参数：无

    返回：
        - en_list = [['I', 'have', 'a', 'dog', '.'], [...]]
        - zh_list = [['我', '有', '一只', '小狗', '。'], [...]]       
    '''

    path = 'D:/科研/python代码/炼丹手册/数据集/VATEX/data/'
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
        print(f'示例：{en_list[0]}\t{zh_list[0]}')
        print(f'示例：{en_list[-1]}\t{zh_list[-1]}')
        print('加载数据集[VATEX]完成！')
        return en_list, zh_list


def load_NMT():
    '''读取NMT数据集，对文本进行分词，以列表返回
    参数：无

    返回：
        - en_list = [['I', 'have', 'a', 'dog', '.'], [...]], 
        - zh_list = [['我', '有', '一只', '小狗', '。'], [...]], 
    '''

    path = "D:/科研/python代码/炼丹手册/数据集/NMT/en-cn/"
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
        print(f'示例：{en_list[0]}\t{zh_list[0]}')
        print(f'示例：{en_list[-1]}\t{zh_list[-1]}')
        print(f'加载数据集[NMT]完成！')
        return en_list, zh_list


def build_dic(sent_list, vocab_size, S='BOS', E='EOS', P='PAD', U='UNK', case_sensitive=False):
    '''按出现频次，构建词表，key：单词，value：id
    参数：
        - sent_list：分词后的句子列表，例如：[['I', 'love', 'you', '.'], [...], ...], 
        - vocab_size：字典大小，容纳的单词数量
        - S：'BOS' 起始标志 
        - E：'EOS' 结束标志 
        - P：'PAD' 填充标志 
        - U：'UNK' 未知标志
        - case_sensitive：False，区分大小写，例如：'the' 和 'The'

    返回：vocab_dict：单词表       
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
    print(f'语料单词总数：{len(word_count)}  最低词频：{ls[-1][-1]}  字典大小：{vocab_size}')

    for i, w in enumerate(ls):
        w = w[0]
        vocab_dict[w] = i + 4
        if not case_sensitive:
            w = _dic[w[0]]+w[1:] if w[0] in _dic.keys() else w
            vocab_dict[w] = i + 4
    # vocab_dict.update({w[0]: i + 4 for i, w in enumerate(ls)})
    return vocab_dict


def sent2vec(sent_list, vec_len, vocab_dict, S='BOS', E='EOS', P='PAD', U='UNK'):
    '''将句子按照字典映射为向量
    参数：
        - sent_list: 分词后的句子列表，例如：[['I', 'love', 'you', '.'], [...], ...], 
        - len：向量长度，句子超出该长度则截断，不足则补齐
        - vocab_dict：词表
        - S：'BOS' 起始标志，None时则表示不添加起始标志
        - E：'EOS' 结束标志，None时则表示不添加结束标志
        - P：'PAD' 填充标志 
        - U：'UNK' 未知标志
    返回：
        - vec_list：向量化之后的句子列表，例如：[['1', '83', '72', '64', 2, 0, 0, ...], [...], ...], 
    '''

    # 添加'BOS''EOS'开始标志
    if S: sent_list = [[S]+sent for sent in sent_list]
    if E: sent_list = [sent+[E] for sent in sent_list]
    # 添加'PAD'填充标志/截断长句子
    sent_list = [sent+[P]*(vec_len-len(sent)) if len(sent)<vec_len else sent[:vec_len] for sent in sent_list]
    # 生成vec_list
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
    '''加载训练数据
    加载分词后的文本列表，生成训练数据。由于transformer的标签以
    one-hot表示，数据占用很大，所以采用data generater形式。
    参数：
         - data：[中文列表, 英文列表]（分词后）
         - vocab_dics：[中文词表, 英文词表]
         - vec_len：句向量长度
         - batch_size：数据生成器每次生成的样本量
         - shuffle=True：打乱数据集
         - S = 'BOS' 起始标志，和词表一致
         - E = 'EOS' 结束标志，和词表一致
         - P = 'PAD' 填充标志，和词表一致
         - U = 'UNK' 未知标志，和词表一致
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
    '''zh-en中文翻译成英文测试
    参数：
        - zh_txt_list：中文文本列表, ['我有一只小狗。', '它非常可爱！', ...]
        - max_len：模型输入向量长度
        - [zh_vocab_dict, en_vocab_dict],中英文单词表
        - model：transformer模型
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
    """回调器，测试模型
    参数：
        - model：训练的模型
        - zh_txt_list：测试中文文本
        - max_len：句向量最大长度
        - vocab_dicts：中英文词表
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
        '''自定义学习率
        实现对学习率的warm_up调整
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
    '''自定义损失函数
    由于目标序列是填充（padded）过的，
    因此在计算损失函数时，应用填充遮挡非常重要。
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