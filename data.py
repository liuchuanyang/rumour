# import torch
import pandas as pd
# 百度分词
from LAC import LAC 
import sys
from tqdm import tqdm
import datetime
def flush():
    sys.stdout.flush()
def read_data(files):
    data=[]
    df=pd.read_csv(files)
    # print(pd.head())
    stop_words_list=stop_words()
    start_time=datetime.datetime.now()
    for _, row in df.iterrows():
        # print(index, row)
        # 分词
        text_list=seg_word(row["text"])
        # print(text_list)
        tmp=[]
        # 去停用词
        for word in text_list:
            if word not in stop_words_list:
                # print(word)
                tmp.append(word)
        # text_list=tmp
        label=row["label"]
        data.append([tmp, label])
    endtime=datetime.datetime.now()
    print((endtime-start_time).minute)
    return data
# 读取停用词
def stop_words():
    stop_words_list=open("stop_words.txt", "r+", encoding="utf-8").read().split("\n")
    # print(data_list)
    return stop_words_list
def remove_stop_words():
    pass

# 分词
def seg_word(strings):
    lac=LAC(mode="seg")
    return lac.run(strings)
def test(files):
    stop_words_list=stop_words()
    data=[]
    df=pd.read_csv(files)
    start_time=datetime.datetime.now()
    df["seg"]=df["text"].apply(lambda x:seg_word(x))
    endtime=datetime.datetime.now()
    print((endtime-start_time).minute)
    # print(df["seg"])
if __name__=="__main__":
    data=read_data("./data/train.csv")
    # print(data[0:2])
    # print(seg_word("我是中国人"))
    # stop_words_list()
    # test("./data/train.csv")