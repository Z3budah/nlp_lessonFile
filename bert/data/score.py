import os

count=0
ori_label=[]
text_label=[]
with open("validation.txt", "r", encoding="UTF-8") as f:
    with open("result.txt", "r", encoding="UTF-8") as f1:
        next(f)
        next(f1)
        lines = f.readlines()
        labels=f1.readlines()
        for line,label in zip(lines,labels):
            count+=1
            print(count)
            text_l=label[0].strip()
            ori_l=line[0].strip()
            # text=line[2:].strip()
            # print(text_l+' '+ori_l+' '+text)
            text_label.append(int(text_l))
            ori_label.append(int(ori_l))

        from sklearn.metrics import precision_score, recall_score, f1_score
        f1score=f1_score(text_label, ori_label, average='binary')
        precision=precision_score(text_label, ori_label)
        recall=recall_score(text_label, ori_label)
        print('f1_score='+str(f1score))
        print('precision_score=' + str(precision))
        print('recall_score=' + str(recall))

    f1.close()
f.close()
