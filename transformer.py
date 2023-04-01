# -*- coding: gbk -*-
from keras.layers import Layer, LayerNormalization, Input, Dense
from keras.models import Model
from keras import backend as K
from keras.activations import leaky_relu
import numpy as np
import tensorflow as tf


class Embedding(Layer):
    '''词嵌入层，该层接受一个batch的句子样本，每个句子由向量表示，向量长度固定，值为每个单词对应单词表的索引值，
    该层将句子样本转换成embedding
    参数：
        model_dim：词的嵌入维度，将索引值转换为一个长度为model_dim的向量
        vocab_size：单词表的大小，需要将所有单词的索引包括起来
        trainable:参数是否可训练

    输入：句子向量，维度（batch, seq_len），seq_len为句子长度
    输出：embedding，维度（batch, seq_len, model_dim）
    '''

    def __init__(self, model_dim, vocab_size, is_trainable=False, **kwargs):
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.is_trainable = is_trainable
        super(Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embeddings = self.add_weight(name="embeddings",
                                           shape=(self.vocab_size, self.model_dim),
                                           initializer='glorot_uniform',
                                           trainable=self.is_trainable)
        super(Embedding, self).build(input_shape)

    def call(self, inputs):    
        if K.dtype(inputs) != 'int32': inputs = K.cast(inputs, 'int32')
        embedding = K.gather(self.embeddings, inputs)
        return embedding * tf.math.sqrt(tf.cast(self.model_dim, tf.float32))

    def compute_output_shape(self, input_shape):
        return (input_shape[-1], self.model_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_dim': self.model_dim,
            'vocab_size': self.vocab_size,
            'is_trainable': self.is_trainable
            })
        return config


class PositionalEncoding(Layer):
    '''位置编码层，该层接受句子的embedding，为其添加位置信息
    参数：无

    输入：embedding，维度（batch, seq_len, model_dim）
    输出：positional_encoding，维度（batch, seq_len, model_dim）
    '''

    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = K.int_shape(inputs)[1]
        model_dim = K.int_shape(inputs)[2]
        positional_encoding = np.zeros((seq_length, model_dim))
        for pos in range(seq_length):
            for i in range(model_dim):
                positional_encoding[pos][i] = pos / np.power(10000, (i-i%2) / model_dim)
        positional_encoding[:, 0::2] = np.sin(positional_encoding[:, 0::2])
        positional_encoding[:, 1::2] = np.cos(positional_encoding[:, 1::2])
        return K.cast(positional_encoding, 'float32')

    def compute_output_shape(self, input_shape):
        return input_shape


class Add(Layer):
    '''相加层，将两个张量相加
    参数：无

    输入：张量构成的数组[张量1, 张量2]
    输出：融合位置信息的embedding
    '''

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def call(self, inputs):
        input_a, input_b = inputs
        res = input_a + input_b
        return res

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class ScaledDotProductAttention(Layer):
    '''缩放点积层，根据qkv和mask获得注意力值
    参数：dropout_rate：随机丢失概率

    输入：[q矩阵，k矩阵，v矩阵，mask矩阵]
    输出：attention value矩阵，和embeddings维度相同，即（batch, seq_len, model_dim）
    '''

    def __init__(self, dropout_rate=0.1, **kwargs):
        self.dropout_rate = dropout_rate
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    def call(self, inputs):
        q, k, v, mask = inputs
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        if K.dtype(q) != 'float32': q = K.cast(q, 'float32')
        if K.dtype(k) != 'float32': k = K.cast(k, 'float32')
        if K.dtype(v) != 'float32': v = K.cast(v, 'float32')
        qk = tf.matmul(q,k,transpose_b=True) / tf.sqrt(dk)
        mask = K.tile(mask, [K.shape(qk)[0]//K.shape(mask)[0], 1,1])
        qk = qk + mask
        atten_value = K.softmax(qk)
        qkv = tf.matmul(atten_value, v)
        qkv = K.dropout(qkv, self.dropout_rate)
        return qkv

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'dropout_rate': self.dropout_rate
        })
        return config


class MultiHeadAttention(Layer):
    '''多头注意力层，计算融合多个特征空间的attention value
    参数：
        n_heads：注意力头数
        model_dim：模型的维度
        dropout_rate：丢失率
        is_trainable：参数是否可训练，默认为True

    输入：[q矩阵，k矩阵，v矩阵，mask矩阵]
    输出：融合多头注意力信息的输出向量，和enbedding维度一致
    '''

    def __init__(self, n_heads, model_dim, dropout_rate, is_trainable=True, **kwargs):
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.dropout_rate = dropout_rate
        self.is_trainable = is_trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w_q = self.add_weight(
            name='w_q',
            shape=(input_shape[0][-1], self.model_dim),
            initializer='glorot_uniform',
            trainable=self.is_trainable)
        self.w_k = self.add_weight(
            name='w_k',
            shape=(input_shape[1][-1], self.model_dim),
            initializer='glorot_uniform',
            trainable=self.is_trainable)
        self.w_v = self.add_weight(
            name='w_v',
            shape=(input_shape[2][-1], self.model_dim),
            initializer='glorot_uniform',
            trainable=self.is_trainable)
        self.w_c = self.add_weight(
            name='w_0',
            shape=(self.model_dim, input_shape[0][-1]),
            initializer='glorot_uniform',
            trainable=self.is_trainable)
        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs):

        q, k, v, mask = inputs
        q_linear = K.dot(q, self.w_q)
        k_linear = K.dot(k, self.w_k)
        v_linear = K.dot(v, self.w_v)
        q_multi_heads = tf.concat(tf.split(q_linear, self.n_heads, axis=2), axis=0)
        k_multi_heads = tf.concat(tf.split(k_linear, self.n_heads, axis=2), axis=0)
        v_multi_heads = tf.concat(tf.split(v_linear, self.n_heads, axis=2), axis=0)
        att_outputs = ScaledDotProductAttention(
            dropout_rate = self.dropout_rate
            )([q_multi_heads, k_multi_heads, v_multi_heads, mask])
        att_outputs = tf.concat(tf.split(att_outputs, self.n_heads, axis=0), axis=2)
        outputs = K.dot(att_outputs, self.w_c)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_heads': self.n_heads,
            'model_dim': self.model_dim,
            "dropout_rate" : self.dropout_rate,
            "is_trainable" : self.is_trainable
            })
        return config


class FeedForward(Layer):
    '''前向传递层，将样本矩阵映射到新的维度，然后再映射回来
    参数：
        inner_dim：线性映射维度
        is_trainable：参数是否可训练

    输入：基于多头注意力层输出的句子样本矩阵，维度（batch, seq_len, model_dim）和embedding一致
    输出：投影变换后的样本矩阵，维度不变
    '''

    def __init__(self, inner_dim, is_trainable=True, dropout_rate=0.1, **kwargs):
        self.inner_dim = inner_dim
        self.is_trainable = is_trainable
        self.dropout_rate = dropout_rate
        super(FeedForward, self).__init__(**kwargs)

    def build(self, input_shape):
        model_dim = input_shape[-1]
        self.weights_inner = self.add_weight(
            name="weights_inner",
            shape=(model_dim, self.inner_dim),
            initializer='glorot_uniform',
            trainable=self.is_trainable)
        self.weights_out = self.add_weight(
            name="weights_out",
            shape=(self.inner_dim, model_dim),
            initializer='glorot_uniform',
            trainable=self.is_trainable)
        self.bais_inner = self.add_weight(
            name="bais_inner",
            shape=(self.inner_dim,),
            initializer='uniform',
            trainable=self.is_trainable)
        self.bais_out = self.add_weight(
            name="bais_out",
            shape=(model_dim,),
            initializer='uniform',
            trainable=self.is_trainable)
        super(FeedForward, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'float32': inputs = K.cast(inputs, 'float32')
        inner_out = leaky_relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bais_out
        outputs = K.dropout(outputs, self.dropout_rate)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            'inner_dim': self.inner_dim,
            'is_trainable': self.is_trainable,
            'dropout_rate': self.dropout_rate})
        return config


class EncoderBlock(Layer):
    '''编码块，transformer中的编码器由多个编码块堆叠而成
    参数：
        n_heads：注意力头的数量
        model_dim：模型维度
        dropout_rate：丢失率
        inner_dim：前向传递层中的线性投影维度

    输入：[句子样本矩阵, paading_mask掩码矩阵]，样本矩阵维度和embedding一致
    输出：经过多头注意力层和前向传递层处理后的句子样本矩阵，维度保持不变
    '''

    def __init__(self, n_heads, model_dim, dropout_rate, inner_dim, **kwargs): 
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.dropout_rate = dropout_rate
        self.inner_dim = inner_dim
        self.MultiHeadAttention = MultiHeadAttention(n_heads=self.n_heads,
                                                     model_dim=self.model_dim,
                                                     dropout_rate=self.dropout_rate)
        self.Add = Add()
        self.layer_norm = LayerNormalization()
        self.FeedForward = FeedForward(inner_dim=self.inner_dim, dropout_rate = self.dropout_rate)
        super(EncoderBlock, self).__init__(**kwargs)

    def call(self,inputs):
        inputs, mask = inputs
        attention_out = self.MultiHeadAttention([inputs, inputs, inputs, mask])
        attention_out = self.Add([inputs, attention_out])
        attention_out = self.layer_norm(attention_out)
        ff_out = self.FeedForward(attention_out)
        ff_out = self.Add([attention_out, ff_out])
        ff_out = self.layer_norm(ff_out)
        return ff_out

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_heads': self.n_heads,
            'model_dim': self.model_dim,
            'dropout_rate': self.dropout_rate,
            'inner_dim': self.inner_dim
            })
        return config


class DecoderBlock(Layer):
    '''解码块，transformer中的解码器由多个解码块堆叠而成
    参数：
        n_heads：注意力头的数量
        model_dim：模型维度
        droupout_rate：丢失率
        inner_dim：前向传递层中的线性投影维度

    输入：
        目标句子样本矩阵或前一个DecoderBlock的输出矩阵，EncoderBlocks输出的矩阵，掩码矩阵，训练掩码矩阵
        即数组[decoder_input, encoder_output, padding_mask, sequence_mask]
    输出：经过多头注意力层和前向传递层处理后的句子样本矩阵，维度保持不变
    '''

    def __init__(self, n_heads, model_dim, dropout_rate, inner_dim, **kwargs):
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.dropout_rate = dropout_rate
        self.inner_dim = inner_dim
        self.Masked_MultiHeadAttention = MultiHeadAttention(n_heads=self.n_heads,
                                                            model_dim=self.model_dim,
                                                            dropout_rate=self.dropout_rate)
        self.MultiHeadAttention = MultiHeadAttention(n_heads=self.n_heads,
                                                     model_dim=self.model_dim,
                                                     dropout_rate=self.dropout_rate)
        self.Add = Add()
        self.layer_norm = LayerNormalization()
        self.FeedForward = FeedForward(inner_dim=self.inner_dim, dropout_rate = self.dropout_rate)
        super(DecoderBlock, self).__init__(**kwargs)

    def call(self,inputs):
        decoder_input, encoder_output, padding_mask, sequence_mask = inputs
        masked_attention_out = self.Masked_MultiHeadAttention([decoder_input, decoder_input, decoder_input, sequence_mask])
        masked_attention_out = self.Add([decoder_input, masked_attention_out])
        masked_attention_out = self.layer_norm(masked_attention_out)
        attention_out = self.MultiHeadAttention([masked_attention_out, encoder_output, encoder_output, padding_mask])
        attention_out = self.Add([masked_attention_out, attention_out])
        attention_out = self.layer_norm(attention_out)
        ff_out = self.FeedForward(attention_out)
        ff_out = self.Add([attention_out, ff_out])
        ff_out = self.layer_norm(ff_out)
        return ff_out

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_heads': self.n_heads,
            'model_dim': self.model_dim,
            'dropout_rate': self.dropout_rate,
            'inner_dim': self.inner_dim
        })
        return config


class Encoder(Layer):
    '''编码器，通过多个编码块EncoderBlock堆叠构成
    参数：
        n_blocks：编码块堆叠个数
        n_heads：编码块中多头注意力层的头数量
        model_dim：编码块中多头注意力层模型维度
        droupout_rate：编码块中多头注意力层的丢失率
        inner_dim：编码块中前向传递层中的线性投影维度

    输入：[样本的embedding, padding_mask]，维度为（batch, seq_len, model_dim）
    输出：经过多个EncoderBlock传递输出的样本句子矩阵
    '''

    def __init__(self, 
                 n_blocks, 
                 n_heads, 
                 model_dim, 
                 dropout_rate, 
                 inner_dim, 
                 **kwargs):
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.dropout_rate = dropout_rate
        self.inner_dim = inner_dim
        self.encoder_blocks = [EncoderBlock(n_heads = self.n_heads,
                                            model_dim = self.model_dim,
                                            dropout_rate = self.dropout_rate,
                                            inner_dim = self.inner_dim) for _ in range(self.n_blocks)]
        super(Encoder, self).__init__(**kwargs)

    def call(self, inputs):
        inputs, padding_mask = inputs
        for i in range(self.n_blocks):
            inputs = self.encoder_blocks[i]([inputs, padding_mask])
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_blocks': self.n_blocks,
            'n_heads': self.n_heads,
            'model_dim': self.model_dim,
            'dropout_rate': self.dropout_rate,
            'inner_dim': self.inner_dim
        })
        return config


class Decoder(Layer):
    '''解码器，通过多个解码块DecoderBlock堆叠构成
    参数：
        n_blocks：解码块堆叠个数
        n_heads：解码块中多头注意力层的头数量
        model_dim：解码块中多头注意力层模型维度
        droupout_rate：解码块中多头注意力层的丢失率
        inner_dim：解码块中FeedForward的线性投影维度

    输入：
        [target_output, encoder_output, padding_mask, sequence_mask]
        当前时刻的目标句子样本矩阵，Encoder输出的矩阵，填充掩码，序列掩码
    输出：经过多个DecoderBlock传递输出下个时刻的单词矩阵
    '''

    def __init__(self,
                 n_blocks, 
                 n_heads, 
                 model_dim, 
                 dropout_rate, 
                 inner_dim, 
                 **kwargs):
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.model_dim = model_dim
        self.dropout_rate = dropout_rate
        self.inner_dim = inner_dim
        self.decoder_blocks = [DecoderBlock(n_heads = self.n_heads,
                                            model_dim = self.model_dim,
                                            dropout_rate = self.dropout_rate,
                                            inner_dim = self.inner_dim) for _ in range(self.n_blocks)]
        super(Decoder, self).__init__(**kwargs)

    def call(self,inputs):
        target_output, encoder_output, padding_mask, sequence_mask = inputs
        for i in range(self.n_blocks):
            target_output = self.decoder_blocks[i]([target_output, encoder_output, padding_mask, sequence_mask])
        return target_output

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_blocks': self.n_blocks,
            'n_heads': self.n_heads,
            'model_dim': self.model_dim,
            'dropout_rate': self.dropout_rate,
            'inner_dim': self.inner_dim
        })
        return config


class Transformer(Layer):
    '''组合Encoder和Decoder，构成完整的transformer模型
    参数：
        - vocab_size：词表大小
        - model_dim：模型维度，需要是n_heads的整数倍
        - n_blocks：编解码器中模块堆叠数
        - n_heads：注意力头数，需要能将model_dim整除
        - inner_dim：编解码器FeedForward投影维度
        - dropout_rate：编码块或解码块的丢失率
        - masking_num：掩码数字，用于降低attention value
        - embedding_trainable：嵌入层参数是否可训练
        
    输入：
        - [encoder input, decoder input]，编码器和解码器的输入矩阵
        - 它们分别为输入句子和目标句子的向量（embedding之前）
    输出：
        预测目标句子在下一时刻的所有单词softmax概率，one-hot向量

    transform在两个地方使用了权重共享：
    （1）Encoder和Decoder间的Embedding层权重共享；
    （2）Decoder中Embedding层和FC层权重共享。
    multi-head attention中每个head要降维，在更低的维度，
    在多个独立的特征空间，更容易学习到更丰富的特征信息
    '''

    def __init__(self,
                 vocab_size=5000, 
                 model_dim=256,   
                 n_blocks=6,
                 n_heads=4, 
                 inner_dim=256,
                 dropout_rate=0.1,
                 masking_num=-2**32+1,
                 embedding_trainable=True,
                 **kwargs):

        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.inner_dim = inner_dim
        self.dropout_rate = dropout_rate
        self.masking_num = masking_num
        self.embedding_trainable = embedding_trainable

        self.encoder_embedding = Embedding(model_dim=self.model_dim,
                                   vocab_size=self.vocab_size,
                                   is_trainable=self.embedding_trainable)
        self.decoder_embedding = Embedding(model_dim=self.model_dim,
                                  vocab_size=self.vocab_size,
                                  is_trainable=self.embedding_trainable)

        self.encoder = Encoder(n_blocks=self.n_blocks,
                               n_heads=self.n_heads,
                               model_dim=self.model_dim,
                               dropout_rate=self.dropout_rate,
                               inner_dim=self.inner_dim)
        self.decoder = Decoder(n_blocks=self.n_blocks,
                               n_heads=self.n_heads,
                               model_dim=self.model_dim,
                               dropout_rate=self.dropout_rate,
                               inner_dim=self.inner_dim)

        self.out_layer = Dense(self.vocab_size)
        super(Transformer, self).__init__(**kwargs)

    def padding_mask(self, inputs):
        masks = K.equal(inputs, 0)
        masks = K.expand_dims(masks, axis=1)
        masks = K.tile(masks, [1, K.shape(inputs)[-1], 1]) 
        masks = K.cast(masks, 'float32')
        return masks * self.masking_num

    def sequence_mask(self, inputs):
        qk = K.expand_dims(inputs, axis=-1)
        qk = K.tile(qk, [1, 1, K.shape(inputs)[-1]]) 
        seq_mask = 1-tf.linalg.band_part(tf.ones_like(qk), -1, 0)
        seq_mask = K.cast(seq_mask, 'float32')
        return seq_mask * self.masking_num

    def call(self, inputs):
        # 编码器和解码器的输入
        encoder_input, decoder_input = inputs
        # padding_mask和sequence_mask
        encoder_mask = self.padding_mask(encoder_input)
        decoder_mask = self.padding_mask(decoder_input)
        seq_mask = self.sequence_mask(decoder_input)

        # embedding
        encoder_input = self.encoder_embedding(encoder_input)
        decoder_input = self.decoder_embedding(decoder_input)

        # embedding + positionalencoding
        encoder_input = Add()([encoder_input, PositionalEncoding()(encoder_input)])
        decoder_input = Add()([decoder_input, PositionalEncoding()(decoder_input)])

        # 编码器和解码器的输出
        encoder_output = self.encoder([encoder_input, encoder_mask])
        decoder_output = self.decoder([decoder_input, encoder_output, decoder_mask, seq_mask])
        # 计算softmax输出
        # linear_projection = K.dot(decoder_output, K.transpose(self.encoder_embedding.embeddings))
        linear_projection = self.out_layer(decoder_output)
        outputs = K.softmax(linear_projection)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'n_blocks': self.n_blocks,
            'n_heads': self.n_heads, 
            'model_dim': self.model_dim,
            'inner_dim': self.inner_dim,
            'dropout_rate': self.dropout_rate,
            'masking_num': self.masking_num,
            'embedding_trainable': self.embedding_trainable
        })
        return config


class my_trainsformer(tf.keras.Model):
    def __init__(self,
                 vocab_size=5000, 
                 model_dim=256,   
                 n_blocks=6,
                 n_heads=4, 
                 inner_dim=256,
                 dropout_rate=0.1,
                 masking_num=-2**32+1,
                 embedding_trainable=True,
                 **kwargs):
        super(my_trainsformer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.inner_dim = inner_dim
        self.dropout_rate = dropout_rate
        self.masking_num = masking_num
        self.embedding_trainable = embedding_trainable
        
        self.encoder_embedding = Embedding(model_dim=self.model_dim,
                                           vocab_size=self.vocab_size,
                                           is_trainable=self.embedding_trainable)

        self.decoder_embedding = Embedding(model_dim=self.model_dim,
                                           vocab_size=self.vocab_size,
                                           is_trainable=self.embedding_trainable)

        self.encoder = Encoder(n_blocks=self.n_blocks,
                               n_heads=self.n_heads,
                               model_dim=self.model_dim,
                               dropout_rate=self.dropout_rate,
                               inner_dim=self.inner_dim)
        self.decoder = Decoder(n_blocks=self.n_blocks,
                               n_heads=self.n_heads,
                               model_dim=self.model_dim,
                               dropout_rate=self.dropout_rate,
                               inner_dim=self.inner_dim)

        self.out_layer = Dense(self.vocab_size)

    def padding_mask(self, inputs):
        masks = K.equal(inputs, 0)
        masks = K.cast(masks, 'float32')
        masks = K.expand_dims(masks, axis=-1)
        masks = K.tile(masks, [1, 1, K.shape(inputs)[-1]])       
        return masks * self.masking_num

    def sequence_mask(self, inputs):
        qk = K.expand_dims(inputs, axis=-1)
        qk = K.tile(qk, [1, 1, K.shape(inputs)[-1]]) 
        seq_mask = 1-tf.linalg.band_part(tf.ones_like(qk), -1, 0)
        seq_mask = K.cast(seq_mask, 'float32')
        return seq_mask * self.masking_num

    def call(self, inputs):
        # 编码器和解码器的输入
        encoder_input, decoder_input = inputs
        # padding_mask和sequence_mask
        encoder_mask = self.padding_mask(encoder_input)
        decoder_mask = self.padding_mask(decoder_input)
        seq_mask = self.sequence_mask(decoder_input)
        # embedding
        encoder_input = self.encoder_embedding(encoder_input)
        decoder_input = self.decoder_embedding(decoder_input)
        # embedding + positionalencoding
        encoder_input = Add()([encoder_input, PositionalEncoding()(encoder_input)])
        decoder_input = Add()([decoder_input, PositionalEncoding()(decoder_input)])
        # 编码器和解码器的输出
        encoder_output = self.encoder([encoder_input, encoder_mask])
        decoder_output = self.decoder([decoder_input, encoder_output, decoder_mask, seq_mask])
        # 计算softmax输出
        linear_projection = self.out_layer(decoder_output)
        outputs = K.softmax(linear_projection)
        return outputs


def build_model(vocab_size=5000, 
                max_seq_len = 50,
                model_dim=128,   
                n_blocks=6,
                n_heads=4, 
                inner_dim=256,
                dropout_rate=0.1,
                masking_num=-2**32+1,
                embedding_trainable=True):

    encoder_input = Input((max_seq_len, ))
    decoder_input = Input((max_seq_len, ))
    decoder_output = Transformer(
        vocab_size,                            
        model_dim, 
        n_blocks, 
        n_heads, 
        inner_dim, 
        dropout_rate,
        masking_num,
        embedding_trainable
        )([encoder_input,decoder_input])
    model = Model([encoder_input,decoder_input],decoder_output)
    return model