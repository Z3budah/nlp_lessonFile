
import numpy as np
# import matplotlib.pyplot as plt
import re
import jieba
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")
'''
wv_model = KeyedVectors.load_word2vec_format('embeddings/sgns.zhihu.bigram',
                                             binary=False, unicode_errors="ignore")
'''
import os

wv_model = KeyedVectors.load_word2vec_format('mymodel.txt',binary = False)


train_texts_orig = []
train_target = []

with open("train.txt", "r", encoding="UTF-8") as f:
    next(f)
    lines = f.readlines()
    for line in lines:

        # print(line[7:len(line)-2].strip())
        train_texts_orig.append(line[2:len(line)-1].strip())
        train_target.append(int(line[0]))

'''
print(train_texts_orig)
print(train_target)
print(len(train_texts_orig))
'''

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow import keras


jieba.load_userdict("userdict.txt")
# 文字字符替换，不属于系统字符
_NUM = "_NUM"
def stopwordlist():
    stopwords=[line.strip() for line in open('stop_words.txt',encoding='UTF-8').readlines()]
    return stopwords

train_tokens=[]
for text in train_texts_orig:
    sentence = ''
    stopwords = stopwordlist()
    for word in text:
        if word not in stopwords:
            if word!='\t':
                sentence+=word
    # print(sentence)

    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
    text = re.sub('\d+', _NUM, text)
    cut=jieba.cut(text)
    # print("/".join(cut))
    cut_list = [ i for i in cut ]
    for i, word in enumerate(cut_list):
        try:
            # 将词转换为索引index
            cut_list[i] = wv_model.vocab[word].index
        except KeyError:
            # 如果词不在字典中，则输出0
            cut_list[i] = 0
    train_tokens.append(cut_list)

# 获得所有tokens的长度
num_tokens=[ len(tokens) for tokens in train_tokens]
num_tokens=np.array(num_tokens)
print(num_tokens)
# 平均tokens的长度
print(np.mean(num_tokens))
# 最长的tokens的长度
print(np.max(num_tokens))
'''
plt.hist(np.log(num_tokens), bins = 100)
plt.xlim((0,5))
plt.ylabel('number of tokens')
plt.xlabel('length of tokens')
plt.title('Distribution of tokens length')
plt.show()
'''
# 取tokens平均值并加上两个tokens的标准差，
# 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖91%左右的样本
# max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
# max_tokens = int(max_tokens)
max_tokens=np.max(num_tokens)
print(max_tokens)
print(np.sum( num_tokens < max_tokens ) / len(num_tokens))

def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + wv_model.index2word[i]
        else:
            text = text + ' '
    return text

embedding_dim = wv_model['元宝'].shape[0]
num_words = 7047

# embedding_dim = wv_model['山东大学'].shape[0]
# num_words = 5000
# 初始化embedding_matrix，之后在keras上进行应用
embedding_matrix = np.zeros((num_words, embedding_dim))
# embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
# 维度为 7047 * 100
for i in range(num_words):
    embedding_matrix[i,:] = wv_model[wv_model.index2word[i]]

embedding_matrix = embedding_matrix.astype('float32')
print("类型",embedding_matrix.dtype)
# 检查index是否对应，
# 输出100意义为长度为100的embedding向量一一对应
# print(np.sum( wv_model[wv_model.index2word[333]] == embedding_matrix[333] ))
# embedding_matrix的维度，
# 这个维度为keras的要求，后续会在模型中用到
# print(embedding_matrix.shape)

# 进行padding和truncating， 输入的train_tokens是一个list
# 返回的train_pad是一个numpy array
train_pad = pad_sequences(train_tokens, maxlen=max_tokens,
                            padding='pre', truncating='pre')
train_pad[ train_pad>=num_words ] = 0
print("pad类型",train_pad.dtype)
# print(train_pad[33])

# 准备target向量
train_target = np.array(train_target)
print("target类型",train_target.dtype)
# 进行训练和测试样本的分割
from sklearn.model_selection import train_test_split
# 进行训练和测试样本的分割
X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                    train_target,
                                                    test_size=0.1,
                                                    random_state=12)

# 查看训练样本
print(reverse_tokens(X_train[35]))
print('class: ',y_train[35])

# 用LSTM对样本进行分类
model = Sequential()
# 模型第一层为embedding
model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_tokens,
                    trainable=False))

model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
model.add(LSTM(units=16, return_sequences=False))

model.add(Dense(1, activation='sigmoid'))
# 我们使用adam以0.001的learning rate进行优化
# optimizer = Adam(lr=1e-3)
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

print(model.summary())

# 建立一个权重的存储点
path_checkpoint = 'sentiment_checkpoint.keras'
checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                                      verbose=1, save_weights_only=True,
                                      save_best_only=True)

# 尝试加载已训练模型
try:
    model.load_weights(path_checkpoint)
except Exception as e:
    print(e)
    print("no model exist")

# 定义early stoping如果3个epoch内validation loss没有改善则停止训练
earlystopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# 自动降低learning rate
lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1, min_lr=1e-8, patience=0,
                                       verbose=1)

# 定义callback函数
callbacks = [
    earlystopping,
    checkpoint,
    lr_reduction
]

# 开始训练
model.fit(X_train, y_train,
          validation_split=0.3,
          epochs=10,
          batch_size=128,
          callbacks=callbacks)

result = model.evaluate(X_test, y_test)
print('Accuracy:{0:.2%}'.format(result[1]))


print("Saving model to disk \n")
model_save_path = "model_file_path.h5"
model.save(model_save_path)

from tensorflow.python.keras.models import load_model
model = load_model(model_save_path)


def predict_sentiment(text):
    print(text)
    # 去标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    text = re.sub('\d+', _NUM, text)
    # 分词
    cut = jieba.cut(text)
    cut_list = [ i for i in cut ]
    # tokenize
    for i, word in enumerate(cut_list):
        try:
            cut_list[i] = wv_model.vocab[word].index
            if cut_list[i] >= 50000:
                cut_list[i] = 0
        except KeyError:
            cut_list[i] = 0
    # padding
    tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
                           padding='pre', truncating='pre')

    # 预测
    result = model.predict(x=tokens_pad)
    coef = result[0][0]
    if coef >= 0.5:
        print('广告','output=%.2f'%coef)
    else:
        print('非广告','output=%.2f'%coef)

test_list = [
    '测试文本'
]
for text in test_list:
    predict_sentiment(text)


