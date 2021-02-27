import os
import numpy as np
import re
import jieba
import jieba.analyse
jieba.load_userdict("userdict.txt")

not_list=['杀', '投','演员','推塔','输','人头','打我','打爆','举报','人透','干','nm', 'tm', 'fw', 'laji', 'sb', 'zz', '低能', '全家', '你爸', '尼玛', '鸡', '干嘛', '曰', '草', '艹', '傻', 'sha',
          'Sha','演', '送', '投','我挂机','你挂机','不打','不想玩','演员','我给你送人头','我送人头','我演','我剐','送纫头','我要演','偓挂机','卦机','开挂']
good_list = ['皮肤', '别挂', '别送', '别投', '别演', '不要送', '不要挂', '不要投', '不要演','表演','不投','怀疑','演我','厉害','别给我送','传送','Buff','是演员吧','被演','没说','穿','你挂了','不信','演示',
             '技能','心态','远点','自己','不送','视而不见','不是送','没有','东西','带飞','我们','除了','以为','怎么','不可以投','少送','回家','不想送','不想演','蓝','红','影响','英雄','快递','血','救','坑'
             ,'卡','缤纷','烦','我家','回城','街头霸王','送死','站','比我','跟着','翻盘','跟随','给我','遇见','别再送','不该','大招']
count=0

def write_negpos():
    with open("trainori.txt", "r", encoding="UTF-8") as f:
        next(f)
        lines = f.readlines()
        for line in lines:
            postive = ''
            negative = ''
            for line in lines:
                if (int(line[-2]) == 0):
                    postive += line

                else:
                    negative += line

            with open('train_pos.txt', 'w', encoding="UTF-8") as f2:
                f2.write(postive)
            with open('train_neg.txt', 'w', encoding="UTF-8") as f3:
                f3.write(negative)
    f.close()
    f2.close()
    f3.close()


with open("trainori.txt", "r", encoding="UTF-8") as f:
    count=0
    result=''
    result+=next(f)
    lines = f.readlines()
    for line in lines:
        result+=line[:-2]
        isG=True
        str=line[7:len(line) - 2].strip()
        if (int(line[-2]) == 1):
            result+='1'
            result+='\n'
        else:
            for i in range (len(not_list)):
                if(str.find(not_list[i])>=0):
                    isG=False
                    for j in range(len(good_list)):
                        if (str.find(good_list[j]) >= 0):
                            isG=True
                            break
            if(isG==False):
                result+='1'
                result+='\n'
                print(str)
                count+=1
            else:
                result+='0'
                result+='\n'


    with open('train_new.txt', 'w', encoding="UTF-8") as f2:
        f2.write(result)
print(count)

