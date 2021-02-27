# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
from gensim.models import word2vec
import os
import numpy as np
import re
import jieba
import jieba.analyse
jieba.load_userdict("userdict.txt")

# 文字字符替换，不属于系统字符
_NUM = "_NUM"
count=0

with open("train.txt", "r", encoding="UTF-8") as f:
    lines = f.readlines()
    result=''
    for line in lines[1:]:
        # count+=1
        # if(count>10): break
        # print(line[7:len(line)-2].strip())
        text=line[2:len(line)-1].strip()
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
        text = re.sub('\d+', _NUM, text)
        cut=jieba.cut(text)
        # print('/'.join(cut))
        result+=' '.join(cut)
        result+='\n'
    with open('segment.txt', 'w',encoding="UTF-8") as f2:
        f2.write(result)
f.close()
f2.close()

import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)


sentences = word2vec.LineSentence('segment.txt')

model = word2vec.Word2Vec(sentences, hs=1,min_count=5,window=5,size=100)

print(model['元宝'])
model.wv.save_word2vec_format('mymodel.txt',binary = False)
model = KeyedVectors.load_word2vec_format('mymodel.txt',binary = False)
print(model['元宝'])

model = KeyedVectors.load_word2vec_format('mymodel.txt',binary = False)
req_count = 5
for key in model.wv.similar_by_word('钻石', topn =100):
    if len(key[0])==3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break;

embedding_dim = model['元宝'].shape[0]
# print('词向量的长度为{}'.format(embedding_dim))
# print(model['送人头'])

num_words = 7047
# 初始化embedding_matrix，之后在keras上进行应用
embedding_matrix = np.zeros((num_words, embedding_dim))
# embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
# 维度为 3925 * 100
for i in range(num_words):
    embedding_matrix[i,:] = model[model.index2word[i]]
embedding_matrix = embedding_matrix.astype('float32')
print(np.sum( model[model.index2word[333]] == embedding_matrix[333] ))
print(embedding_matrix.shape)
'''
