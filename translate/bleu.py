from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import re
sum=0.0
count=0
smooth = SmoothingFunction()
with open("Etest.en", "r", encoding="UTF-8") as f:
    with open('Epredict.en', 'r', encoding="UTF-8") as f2:
        with open('Ebleu.txt', 'w', encoding="UTF-8") as f3:
            ref_lines = f.readlines()
            can_lines=f2.readlines()
            for ref,can in zip(ref_lines,can_lines):
                count+=1
                ref=ref.strip()
                ref=re.sub(r"[^a-zA-Z?!’¿]+", " ", ref)
                reference = []
                reference.append(ref.split())
                can=can.strip()
                candidate=can.split()
                # score = sentence_bleu(reference, candidate, smoothing_function=smooth.method1)
                score = corpus_bleu([reference], [candidate], smoothing_function=smooth.method1)
                sum+=score
                print(score)
                f3.write(str(score)+'\n')
            average=sum/count
            print(average)
            f3.write("average :"+str(average) + '\n')
