#coding=utf-8

#################################
# 情感分析之消极言语识别，主程序模板
# file: main_test.py
#################################

import os
from optparse import OptionParser

###################################
# arg_parser： 读取参数列表
###################################
def arg_parser():
    oparser = OptionParser()

    oparser.add_option("-m", "--model_file", dest="model_file", help="输入模型文件 \
            must be: negative.model", default = 'model_file_path.h5')

    oparser.add_option("-d", "--data_file", dest="data_file", help="输入验证集文件 \
            must be: validation_data.txt", default = 'validation_data.txt')

    oparser.add_option("-o", "--out_put", dest="out_put_file", help="输出结果文件 \
			must be: result.txt", default = 'result.txt')

    (options, args) = oparser.parse_args()
    global g_MODEL_FILE
    g_MODEL_FILE = str(options.model_file)

    global g_DATA_FILE
    g_DATA_FILE = str(options.data_file)

    global g_OUT_PUT_FILE
    g_OUT_PUT_FILE = str(options.out_put_file)

###################################
# load_model： 加载模型文件
###################################
def load_model(model_file_name):
	from tensorflow.keras.models import load_model
	model = load_model(model_file_name)
	return model

###################################
# predict： 根据模型预测结果并输出结果文件，文件内容格式为qid\t言语\t标签
###################################
import jieba
jieba.load_userdict("userdict.txt")
import numpy as np
import re
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


def predict(model):
	print("predict start.......")
	###################################
	# 预测逻辑和结果输出，("%d\t%s\t%d", qid, content, predict_label)
	###################################
	count=0
	max_tokens = 20
	wv_model = KeyedVectors.load_word2vec_format('mymodel.txt', binary=False)
	with open(g_DATA_FILE, "r", encoding="UTF-8") as f:
		head=next(f)
		lines = f.readlines()
		resulttext=''
		resulttext+=head
		for line in lines:
			resulttext+=line[:-3]
			text=line[7:len(line) - 3].strip()
			count+=1
			print(count);
			# 去标点
			text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)

			# 分词
			cut = jieba.cut(text)
			cut_list = [i for i in cut]
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
				resulttext += '1'
			else:
				resulttext += '0'
			resulttext +='\n'
	with open(g_OUT_PUT_FILE, 'w', encoding="UTF-8") as f2:
		f2.write(resulttext)
	f.close()
	f2.close()

	print("predict end.......")

	return None;

###################################
# main： 主逻辑
###################################
def main():
	print("main start.....")
	
	if g_MODEL_FILE is not None:
		model = load_model(g_MODEL_FILE)
		predict(model)

	print("main end.....")

	return 0;

if __name__ == '__main__':
	arg_parser()
	main()
