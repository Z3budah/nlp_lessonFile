import jieba
import numpy as np
import re
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")

wv_model = KeyedVectors.load_word2vec_format('mymodel.txt',binary = False)

from tensorflow.python.keras.preprocessing.sequence import pad_sequences


model_path = "model_file_path.h5"
from tensorflow.python.keras.models import load_model
model = load_model(model_path)

max_tokens=102

_NUM="_NUM"


def predict_sentiment(text):
    # print(text)
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
        # print('广告','output=%.2f'%coef)
        return 1
    else:
        # print('非广告','output=%.2f'%coef)
        return 0

'''
text_list=[]
text_id=[]
ori_label=[]
text_label=[]

with open("validation.txt", "r", encoding="UTF-8") as f:
    next(f)
    lines = f.readlines()
    for line in lines:
        print(line[2:len(line)-1].strip())
        text_list.append(line[2:len(line)-1].strip())
        ori_label.append(int(line[0]))
 


for text in text_list:
    text_label.append(predict_sentiment(text))
    # print(text_label)
    # print(ori_label)
'''
count=0
ori_label=[]
text_label=[]
with open("validation.txt", "r", encoding="UTF-8") as f:
    with open("result.txt", "w", encoding="UTF-8") as fw:
        next(f)
        lines = f.readlines()
        for line in lines:
            count+=1
            print(count)
            text=line[2:len(line) - 1].strip()
            ori_label.append(int(line[0]))
            predict=predict_sentiment(text)
            text_label.append(predict)
            fw.write(str(predict)+'\t'+text+'\n')

        from sklearn.metrics import precision_score, recall_score, f1_score
        f1score=f1_score(text_label, ori_label, average='binary')
        precision=precision_score(text_label, ori_label)
        recall=recall_score(text_label, ori_label)
        print(f1score)
        print(precision)
        print(recall)
        fw.write('f1_score='+str(f1score)+'\n')
        fw.write('precision_score=' + str(precision)+'\n')
        fw.write('recall_score=' + str(recall)+'\n')
    fw.close()
f.close()
