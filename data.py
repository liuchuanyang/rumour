import torch
import pandas as pd
# 百度分词
from LAC import LAC 
import sys
from tqdm import tqdm
import datetime
from itertools import chain
import gensim
import torch.nn as nn
import time
# import jieba
def flush():
    sys.stdout.flush()
def read_data(files):
    data=[]
    df=pd.read_csv(files, nrows=5)
    # print(pd.head())
    # 分词
    lac=LAC(mode="seg")
    stop_words_list=stop_words()
    # 计算花费时间
    start_time=datetime.datetime.now()
    for _, row in df.iterrows():
        # print(index, row)
        # 分词
        # text_list=seg_word(row["text"])

        text_list=lac.run(row["text"])
       
        # text_list=jieba.cut(row["text"])
        # # print(text_list)
        tmp=[]
        # # 去停用词
        for word in text_list:
            if word not in stop_words_list:
                # print(word)
                tmp.append(word)
        # text_list=tmp
        label=row["label"]
        data.append([tmp, label])
        endtime=datetime.datetime.now()
    print((endtime-start_time).seconds)
    return data
# 读取停用词
def stop_words():
    stop_words_list=open("stop_words.txt", "r+", encoding="utf-8").read().split("\n")
    # print(data_list)
    return stop_words_list
def remove_stop_words(words):
    for word in text_list:
        if word not in stop_words_list:
            # print(word)
            tmp.append(word)

# 分词
# def seg_word(strings):
#     lac=LAC(mode="seg")
#     return lac.run(strings)
stop_words_list=stop_words()
def func(words):
    # 分词
    lac=LAC(mode="seg")
    global stop_words_list
    # 计算花费时间
    start_time=datetime.datetime.now()
    text_list=lac.run(row["text"])
    
    # text_list=jieba.cut(row["text"])
    # # print(text_list)
    tmp=[]
    # # 去停用词
    for word in text_list:
        if word not in stop_words_list:
            # print(word)
            tmp.append(word)
    # text_list=tmp
    label=row["label"]
    data.append([tmp, label])
    endtime=datetime.datetime.now()
    print((endtime-start_time).seconds)
    return data
class process():
    def __init__(self,files):
        self.lac=LAC(mode="seg")
        self.df=pd.read_csv(files, nrows=5)
        self.get_stop_words()
    def getData(self):
        data=[]
        self.df["seg"]=self.df["text"].apply(self.remove_stop_words)
        for _, row in self.df.iterrows():
            data.append([row["seg"], row["label"]])
        return data
    def remove_stop_words(self, words):
        words_=self.lac.run(words).copy()
        for index, word in enumerate(words_):
            if word in self.stop_words_list:
                del words_[index]
        return words_
    def get_stop_words(self):
        self.stop_words_list=open("stop_words.txt", "r+", encoding="utf-8").read().split("\n")




if __name__=="__main__":
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    p=process("./data/train.csv")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    data=p.getData()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    # print(data[1])
    train_tokenized=[words for words, _ in data]
    print(train_tokenized[0])

    # # 获取数据集
    # data=read_data("./data/train.csv")
    # # train 6 eval 1 test 3
    # length=len(data)
    # tag=int(length/10)
    # print(tag)
    # # 训练集
    # train_data=data[:tag*6]
    # eval_data=data[tag*6:tag*7]
    # test_data=data[tag*7:]
    # train_tokenized=[]
    # # 分词
    # train_tokenized=[words for words, _ in train_data]
    # eval_tokenized=[words for words, _ in train_data]
    # test_tokenized=[words for words, _ in train_data]
    # print(train_tokenized[0:3])
    # # 训练集词组
    # vocab=set(chain(*train_tokenized))
    # vocab_size=len(vocab)
    # # wvmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
    # #                                                        binary=True, encoding='utf-8')
    # word_to_idx  = {word: i+1 for i, word in enumerate(vocab)}
    # word_to_idx['<unk>'] = 0
    # idx_to_word = {i+1: word for i, word in enumerate(vocab)}
    # idx_to_word[0] = '<unk>'
    # 
    # train_features = torch.tensor(pad_samples(encode_samples(train_tokenized, vocab)))
    # train_labels = torch.tensor([score for _, score in train_data])
    # eval_features = torch.tensor(pad_samples(encode_samples(eval_tokenized, vocab)))
    # eval_labels = torch.tensor([score for _, score in eval_data])
    # test_features = torch.tensor(pad_samples(encode_samples(test_tokenized, vocab)))
    # test_labels = torch.tensor([score for _, score in test_data])
    # 权重
    # weight = torch.zeros(vocab_size+1, 300)

    # for i in range(len(wvmodel.index2word)):
    #     try:
    #         index = word_to_idx[wvmodel.index2word[i]]
    #     except:
    #         continue
    # weight[index, :] = torch.from_numpy(wvmodel.get_vector(
    #     idx_to_word[word_to_idx[wvmodel.index2word[i]]]))






