"""
Created on Sat Jun  3 06:00:19 2017

@author: 代码医生 qq群：40016981，公众号：xiangyuejiqiren
@blog：http://blog.csdn.net/lijin6249
"""

import tensorflow as tf
import numpy as np
import os
from six.moves import xrange


_buckets = []
convo_hist_limit = 1
max_source_length = 0
max_target_length = 0


flags = tf.app.flags
FLAGS = flags.FLAGS
datautil = __import__("datautil")
seq2seq_model = __import__("seq2seq_model")
import datautil
import seq2seq_model



tf.reset_default_graph()


_buckets =[(3, 3), (5, 5), (10, 10)]#, (20, 20)]# ["40,10","50,15"]
max_train_data_size= 0#(0: no limit)

data_dir = "datacn/"

dropout = 1.0 
grad_clip = 5.0
batch_size = 60
hidden_size = 512
num_layers =2
learning_rate =0.5
lr_decay_factor =0.99

checkpoint_dir= "data/checkpoints/"



###############翻译
hidden_size = 512
checkpoint_dir= "fanyichina/checkpoints/"
data_dir = "fanyichina/"
_buckets =[(20, 20), (40, 40), (50, 50), (60, 60)]

def getfanyiInfo():
    vocaben, rev_vocaben=datautil.initialize_vocabulary(os.path.join(datautil.data_dir, datautil.vocabulary_fileen))
    vocab_sizeen= len(vocaben)
    print("vocab_size",vocab_sizeen)
    
    vocabch, rev_vocabch=datautil.initialize_vocabulary(os.path.join(datautil.data_dir, datautil.vocabulary_filech))
    vocab_sizech= len(vocabch)
    print("vocab_sizech",vocab_sizech) 

    # return vocab_sizeen,vocab_sizech,vocaben,rev_vocabch
    return vocab_sizeen, vocab_sizech, vocabch, rev_vocaben
################################################################    
#source_train_file_path = os.path.join(datautil.data_dir, "data_source_test.txt")
#target_train_file_path = os.path.join(datautil.data_dir, "data_target_test.txt")    
    

def main():
	
    # vocab_sizeen,vocab_sizech,vocaben,rev_vocabch= getfanyiInfo()
    vocab_sizeen, vocab_sizech, vocabch, rev_vocaben = getfanyiInfo()

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    print ("checkpoint_dir is {0}".format(checkpoint_dir))
    with tf.Session() as sess:

        model = createModel(sess,True,vocab_sizech,vocab_sizeen)
    
        print (_buckets)
        model.batch_size = 1

        conversation_history =[]
        while True:
            with open('Ctest.zh', 'r',encoding="UTF-8") as test_f:
                with open('Epredict.en', 'a',encoding="UTF-8") as pred_f:
                    lines = test_f.readlines()
                    for line in lines[5670:]:
                        sentence=line.strip()
                        print(sentence)
                        # prompt = "请输入: "
                        # sentence = input(prompt)
                        conversation_history.append(sentence.strip())
                        conversation_history = conversation_history[-convo_hist_limit:]
                        token_ids = list(reversed(datautil.sentence_to_ids(" ".join(conversation_history), vocabch, normalize_digits=True,Isch=True)))
                        # token_ids = list(reversed(vocab.tokens2Indices(" ".join(conversation_history))))
                        print(token_ids)
                        # token_ids = list(reversed(vocab.tokens2Indices(sentence)))
                        L=[b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)]
                        if L:
                            bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
                            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]},
                                                                                             bucket_id)
                            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                                                             bucket_id, True)
                            # TODO implement beam search
                            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                            print("outputs", outputs, datautil.EOS_ID)
                            if datautil.EOS_ID in outputs:
                                outputs = outputs[:outputs.index(datautil.EOS_ID)]
                                # print(vocab.indices2Tokens(outputs))
                                # print("结果",datautil.ids2texts(outputs,rev_vocaben))
                                convo_output = " ".join(datautil.ids2texts(outputs, rev_vocaben))
                                conversation_history.append(convo_output)
                                convo_output = convo_output.replace("_UNK", "")
                                print(convo_output+'\n')
                                pred_f.write(convo_output+'\n')
                            else:
                                print("can not translation！\n")
                                pred_f.write("can not translation！" + '\n')
                        else:
                            print("can not translation！\n")
                            pred_f.write("can not translation！" + '\n')
                pred_f.close()
            test_f.close()
            break










def createModel(session, forward_only,from_vocab_size,to_vocab_size):
    """Create translation model and initialize or load parameters in session."""
    model = seq2seq_model.Seq2SeqModel(
      from_vocab_size,#from
      to_vocab_size,#to
      _buckets,
      hidden_size,
      num_layers,
      dropout,
      grad_clip,
      batch_size,
      learning_rate,
      lr_decay_factor,
      forward_only=forward_only,
      dtype=tf.float32)
      
    print("model is ok")

    
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt!=None:
        model.saver.restore(session, ckpt)
        print ("Reading model parameters from {0}".format(ckpt))
    else:
        print ("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())  

    return model




if __name__=="__main__":
	main()
