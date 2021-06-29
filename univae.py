#! -*- coding: utf-8 -*-
# UniVAE参考实现
# 链接：https://kexue.fm/archives/8475

import json
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.layers import Loss, integerize_shape
from bert4keras.models import build_transformer_model, RoFormer
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.snippets import AutoRegressiveDecoder, text_segmentate
from keras.layers import Input, Dense, Lambda, Concatenate, Layer
from keras.models import Model

# 基本信息
maxlen = 32
batch_size = 128
epochs = 10000
kappa = 32
z_dim = 16
num_latent_layers = 4

# 模型路径
config_path = '/root/kg/bert/chinese_roformer-char_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roformer-char_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roformer-char_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def split(text):
    """分割句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen * 1.2, seps, strips)


def corpus():
    """读取语料
    """
    while True:
        f = '/root/data_pretrain/synonyms_shuf.json'
        with open(f) as f:
            for l in f:
                d = json.loads(l)
                text, synonyms = d['text'], d['synonyms']
                text = np.random.permutation([text] + synonyms)[0]
                yield split(text)[0]


class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.some_samples = []

    def __iter__(self, random=False):
        batch_token_ids = []
        for is_end, text in self.sample(random):
            self.some_samples.append(text)
            if len(self.some_samples) > 1000:
                self.some_samples.pop(0)
            token_ids, _ = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids, maxlen)
                zeros = np.zeros_like(batch_token_ids)
                ones = np.ones_like(batch_token_ids)
                batch_segment_ids = np.concatenate([zeros, ones], axis=1)
                batch_token_ids = np.concatenate([batch_token_ids] * 2, axis=1)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids = []


class UniAE_Mask(object):
    """仿UniLM做AE模型
    """
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        if self.attention_bias is None:

            def uniae_mask(s, first=True):
                idxs = K.cumsum(s, axis=1)
                mask1 = K.equal(s[:, None, :], s[:, :, None])
                mask2 = idxs[:, None, :] <= idxs[:, :, None]
                mask = K.cast(mask1 & mask2, K.floatx())
                if first:
                    mask = [K.ones_like(mask[..., :1]), mask[..., 1:]]
                    mask = K.concatenate(mask, axis=2)
                return -(1 - mask[:, None]) * 1e12

            self.attention_bias1 = self.apply(
                inputs=self.inputs[1],
                layer=Lambda,
                function=uniae_mask,
                arguments={'first': False},
                name='Attention-UniAE1-Mask'
            )

            self.attention_bias2 = self.apply(
                inputs=self.inputs[1],
                layer=Lambda,
                function=uniae_mask,
                arguments={'first': True},
                name='Attention-UniAE2-Mask'
            )

            self.attention_bias = [self.attention_bias1, self.attention_bias2]

        if inputs < self.num_hidden_layers - self.num_latent_layers:
            return self.attention_bias[0]
        else:
            return self.attention_bias[1]


class vonMisesFisherSampling(Layer):
    """von Mises Fisher分布重参数
    通过累积概率函数的逆和预计算来实现最简单vMF分布采样
    链接：https://kexue.fm/archives/8404
    """
    def __init__(self, kappa, num_caches=10**7, **kwargs):
        super(vonMisesFisherSampling, self).__init__(**kwargs)
        self.kappa = kappa
        self.num_caches = num_caches

    @integerize_shape
    def build(self, input_shape):
        super(vonMisesFisherSampling, self).build(input_shape)
        self.pw_samples = self.add_weight(
            shape=(self.num_caches,),
            initializer=self.initializer(input_shape[-1]),
            trainable=False,
            name='pw_samples'
        )

    def initializer(self, dims):
        def init(shape, dtype=None):
            x = np.linspace(-1, 1, shape[0] + 2)[1:-1]
            y = self.kappa * x + np.log(1 - x**2) * (dims - 3) / 2
            y = np.cumsum(np.exp(y - y.max()))
            return np.interp((x + 1) / 2, y / y[-1], x)

        return init

    def call(self, inputs):
        mu = inputs
        # 采样w
        idxs = K.random_uniform(
            K.shape(mu[..., :1]), 0, self.num_caches, dtype='int32'
        )
        w = K.gather(self.pw_samples, idxs)
        # 采样z
        eps = K.random_normal(K.shape(mu))
        nu = eps - K.sum(eps * mu, axis=1, keepdims=True) * mu
        nu = K.l2_normalize(nu, axis=-1)
        return w * mu + (1 - w**2)**0.5 * nu

    def get_config(self):
        config = {
            'kappa': self.kappa,
            'num_caches': self.num_caches,
        }
        base_config = super(vonMisesFisherSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class UniVAE(UniAE_Mask, RoFormer):
    """RoFormer + UniAE 做VAE模型
    """
    def __init__(self, *args, **kwargs):
        super(UniVAE, self).__init__(*args, **kwargs)
        self.with_mlm = self.with_mlm or True
        self.num_latent_layers = num_latent_layers
        self.mus = []
        self.mode = 'vae'
        self.z_in = self.apply(
            layer=Input,
            shape=(self.num_latent_layers * z_dim,),
            name='Latent-In'
        )
        self.zs = [None] * (self.num_hidden_layers - self.num_latent_layers)
        self.zs += self.apply(
            inputs=self.z_in,
            layer=Lambda,
            function=lambda x: tf.split(x, self.num_latent_layers, axis=1),
            name='Latent-Split'
        )

    def apply_main_layers(self, inputs, index):
        """在中间层插入隐变量运算
        """
        x = inputs
        if index >= self.num_hidden_layers - self.num_latent_layers:
            z = self.apply(
                inputs=x,
                layer=Lambda,
                function=lambda x: x[:, 0],
                name='CLS-Pooler-%s' % index
            )
            z = self.apply(
                inputs=z,
                layer=Dense,
                units=z_dim,
                kernel_initializer=self.initializer,
                name='In-Projection-%s' % index
            )
            z = self.apply(
                inputs=z,
                layer=Lambda,
                function=lambda z: K.l2_normalize(z, axis=-1),
                name='L2-Normalization-%s' % index
            )
            if self.mode == 'encoder':
                self.mus.append(z)
            if self.mode == 'vae':
                z = self.apply(
                    inputs=z,
                    layer=vonMisesFisherSampling,
                    kappa=kappa,
                    name='ReParameterization'
                )
            if self.mode == 'decoder':
                z = self.zs[index]
            z = self.apply(
                inputs=z,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Out-Projection-%s' % index
            )
            x = self.apply(
                inputs=[x, z],
                layer=Lambda,
                function=lambda xz: K.
                concatenate([xz[1][:, None], xz[0][:, 1:]], axis=1),
                mask=lambda x, m: m[0],
                name='Concatenation-%s' % index
            )
        return super(UniVAE, self).apply_main_layers(x, index)

    def build(self, **kwargs):
        super(UniVAE, self).build(**kwargs)
        self.mode = 'encoder'
        output = self.call(self.model.inputs)
        mu = self.apply(inputs=self.mus, layer=Concatenate, axis=1, name='Mu')
        self.encoder = Model(self.model.inputs, mu)
        self.mode = 'decoder'
        output = self.call(self.model.inputs)
        self.decoder = Model(self.model.inputs + [self.z_in], output)


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# 预训练模型
vae = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model=UniVAE,
    return_keras_model=False
)
model = vae.model
encoder = vae.encoder
decoder = vae.decoder

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(2e-5))
model.summary()


class Vector2Sentence(AutoRegressiveDecoder):
    """隐向量解码为句子
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        z = inputs[0]
        token_ids = np.zeros((output_ids.shape[0], maxlen))
        token_ids[:, 0] = tokenizer._token_start_id
        zeros = np.zeros_like(token_ids)
        ones = np.ones_like(output_ids)
        segment_ids = np.concatenate([zeros, ones], axis=1)
        token_ids = np.concatenate([token_ids, output_ids], axis=1)
        return self.last_token(decoder).predict([token_ids, segment_ids, z])

    def generate(self, z, topk=1):
        z = z.reshape((-1, z_dim))
        z /= (z**2).sum(axis=1, keepdims=True)**0.5
        z = z.reshape(-1)
        output_ids = self.beam_search([z], topk)  # 基于beam search
        return tokenizer.decode(output_ids)


vec2sent = Vector2Sentence(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=maxlen
)


def random_samples(n=3):
    """随机采样重构
    """
    print(u'随机采样效果：')
    for _ in range(n):
        z = np.random.randn(num_latent_layers * z_dim)
        sent = vec2sent.generate(z)
        try:
            print(sent)
        except:
            pass
    print()


def reconstructed_samples(n=3):
    """随机重构效果
    """
    some_samples = train_generator.some_samples
    texts = [np.random.choice(some_samples) for i in range(n)]
    X, S = [], []
    for t in texts:
        x, s = tokenizer.encode(t, maxlen=maxlen)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    print(u'随机重构效果：')
    for i, z in enumerate(Z):
        sent = vec2sent.generate(z)
        try:
            print(u'原句：%s' % texts[i])
            print(u'重构：%s' % sent)
        except:
            pass
    print()


def just_show():
    """随机观察一些样本的效果
    """
    random_samples()
    reconstructed_samples()


class Evaluator(keras.callbacks.Callback):
    """评估模型
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存权重
        model.save_weights('./latest_model.weights')
        # 演示效果
        just_show()


if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(corpus(), batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=1000,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./latest_model.weights')
