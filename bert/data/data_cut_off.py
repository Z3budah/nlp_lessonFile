import os
import pandas as pd
from sklearn.utils import shuffle

pd_all = pd.read_csv('data.txt', sep='\t' )
pd_all = shuffle(pd_all)

dev_set = pd_all.iloc[0:int(pd_all.shape[0]/10)]
train_set = pd_all.iloc[int(pd_all.shape[0]/10): int(pd_all.shape[0])]
dev_set.to_csv("dev.tsv", index=False, sep='\t')
train_set.to_csv("train.tsv", index=False, sep='\t')

pd_all = pd.read_csv('validation.txt', sep='\t' )
pd_all.to_csv("test.tsv", index=False, sep='\t')
