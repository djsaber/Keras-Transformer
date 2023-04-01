# -*- coding: gbk -*-
from keras.layers import Layer, LayerNormalization, Input, Dense
from keras.models import Model
from keras import backend as K
from keras.activations import leaky_relu
import numpy as np
import tensorflow as tf


class Embedding(Layer):
    '''��Ƕ��㣬�ò����һ��batch�ľ���������ÿ��������������ʾ���������ȹ̶���ֵΪÿ�����ʶ�Ӧ���ʱ������ֵ��
    �ò㽫��������ת����embedding
    ������
        model_dim���ʵ�Ƕ��ά�ȣ�������ֵת��Ϊһ������Ϊmodel_dim������
        vocab_size�����ʱ�Ĵ�С����Ҫ�����е��ʵ�������������
        trainable:�����Ƿ��ѵ��

    ���룺����������ά�ȣ�batch, seq_len����seq_lenΪ���ӳ���
    �����embedding��ά�ȣ�batch, seq_len, model_dim��
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
    '''λ�ñ���㣬�ò���ܾ��ӵ�embedding��Ϊ�����λ����Ϣ
    ��������

    ���룺embedding��ά�ȣ�batch, seq_len, model_dim��
    �����positional_encoding��ά�ȣ�batch, seq_len, model_dim��
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
    '''��Ӳ㣬�������������
    ��������

    ���룺�������ɵ�����[����1, ����2]
    ������ں�λ����Ϣ��embedding
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
    '''���ŵ���㣬����qkv��mask���ע����ֵ
    ������dropout_rate�������ʧ����

    ���룺[q����k����v����mask����]
    �����attention value���󣬺�embeddingsά����ͬ������batch, seq_len, model_dim��
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
    '''��ͷע�����㣬�����ں϶�������ռ��attention value
    ������
        n_heads��ע����ͷ��
        model_dim��ģ�͵�ά��
        dropout_rate����ʧ��
        is_trainable�������Ƿ��ѵ����Ĭ��ΪTrue

    ���룺[q����k����v����mask����]
    ������ں϶�ͷע������Ϣ�������������enbeddingά��һ��
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
    '''ǰ�򴫵ݲ㣬����������ӳ�䵽�µ�ά�ȣ�Ȼ����ӳ�����
    ������
        inner_dim������ӳ��ά��
        is_trainable�������Ƿ��ѵ��

    ���룺���ڶ�ͷע����������ľ�����������ά�ȣ�batch, seq_len, model_dim����embeddingһ��
    �����ͶӰ�任�����������ά�Ȳ���
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
    '''����飬transformer�еı������ɶ�������ѵ�����
    ������
        n_heads��ע����ͷ������
        model_dim��ģ��ά��
        dropout_rate����ʧ��
        inner_dim��ǰ�򴫵ݲ��е�����ͶӰά��

    ���룺[������������, paading_mask�������]����������ά�Ⱥ�embeddingһ��
    �����������ͷע�������ǰ�򴫵ݲ㴦���ľ�����������ά�ȱ��ֲ���
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
    '''����飬transformer�еĽ������ɶ�������ѵ�����
    ������
        n_heads��ע����ͷ������
        model_dim��ģ��ά��
        droupout_rate����ʧ��
        inner_dim��ǰ�򴫵ݲ��е�����ͶӰά��

    ���룺
        Ŀ��������������ǰһ��DecoderBlock���������EncoderBlocks����ľ����������ѵ���������
        ������[decoder_input, encoder_output, padding_mask, sequence_mask]
    �����������ͷע�������ǰ�򴫵ݲ㴦���ľ�����������ά�ȱ��ֲ���
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
    '''��������ͨ����������EncoderBlock�ѵ�����
    ������
        n_blocks�������ѵ�����
        n_heads��������ж�ͷע�������ͷ����
        model_dim��������ж�ͷע������ģ��ά��
        droupout_rate��������ж�ͷע������Ķ�ʧ��
        inner_dim���������ǰ�򴫵ݲ��е�����ͶӰά��

    ���룺[������embedding, padding_mask]��ά��Ϊ��batch, seq_len, model_dim��
    ������������EncoderBlock����������������Ӿ���
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
    '''��������ͨ����������DecoderBlock�ѵ�����
    ������
        n_blocks�������ѵ�����
        n_heads��������ж�ͷע�������ͷ����
        model_dim��������ж�ͷע������ģ��ά��
        droupout_rate��������ж�ͷע������Ķ�ʧ��
        inner_dim���������FeedForward������ͶӰά��

    ���룺
        [target_output, encoder_output, padding_mask, sequence_mask]
        ��ǰʱ�̵�Ŀ�������������Encoder����ľ���������룬��������
    ������������DecoderBlock��������¸�ʱ�̵ĵ��ʾ���
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
    '''���Encoder��Decoder������������transformerģ��
    ������
        - vocab_size���ʱ��С
        - model_dim��ģ��ά�ȣ���Ҫ��n_heads��������
        - n_blocks�����������ģ��ѵ���
        - n_heads��ע����ͷ������Ҫ�ܽ�model_dim����
        - inner_dim���������FeedForwardͶӰά��
        - dropout_rate������������Ķ�ʧ��
        - masking_num���������֣����ڽ���attention value
        - embedding_trainable��Ƕ�������Ƿ��ѵ��
        
    ���룺
        - [encoder input, decoder input]���������ͽ��������������
        - ���Ƿֱ�Ϊ������Ӻ�Ŀ����ӵ�������embedding֮ǰ��
    �����
        Ԥ��Ŀ���������һʱ�̵����е���softmax���ʣ�one-hot����

    transform�������ط�ʹ����Ȩ�ع���
    ��1��Encoder��Decoder���Embedding��Ȩ�ع���
    ��2��Decoder��Embedding���FC��Ȩ�ع���
    multi-head attention��ÿ��headҪ��ά���ڸ��͵�ά�ȣ�
    �ڶ�������������ռ䣬������ѧϰ�����ḻ��������Ϣ
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
        # �������ͽ�����������
        encoder_input, decoder_input = inputs
        # padding_mask��sequence_mask
        encoder_mask = self.padding_mask(encoder_input)
        decoder_mask = self.padding_mask(decoder_input)
        seq_mask = self.sequence_mask(decoder_input)

        # embedding
        encoder_input = self.encoder_embedding(encoder_input)
        decoder_input = self.decoder_embedding(decoder_input)

        # embedding + positionalencoding
        encoder_input = Add()([encoder_input, PositionalEncoding()(encoder_input)])
        decoder_input = Add()([decoder_input, PositionalEncoding()(decoder_input)])

        # �������ͽ����������
        encoder_output = self.encoder([encoder_input, encoder_mask])
        decoder_output = self.decoder([decoder_input, encoder_output, decoder_mask, seq_mask])
        # ����softmax���
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
        # �������ͽ�����������
        encoder_input, decoder_input = inputs
        # padding_mask��sequence_mask
        encoder_mask = self.padding_mask(encoder_input)
        decoder_mask = self.padding_mask(decoder_input)
        seq_mask = self.sequence_mask(decoder_input)
        # embedding
        encoder_input = self.encoder_embedding(encoder_input)
        decoder_input = self.decoder_embedding(decoder_input)
        # embedding + positionalencoding
        encoder_input = Add()([encoder_input, PositionalEncoding()(encoder_input)])
        decoder_input = Add()([decoder_input, PositionalEncoding()(decoder_input)])
        # �������ͽ����������
        encoder_output = self.encoder([encoder_input, encoder_mask])
        decoder_output = self.decoder([decoder_input, encoder_output, decoder_mask, seq_mask])
        # ����softmax���
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