from data import *
from model import *

def get_data():
     # 获取数据集
    data=read_data("./data/train.csv")
    # train 6 eval 1 test 3
    length=len(data)
    tag=int(length/10)
    print(tag)
    # 训练集
    train_data=data[:tag*6]
    eval_data=data[tag*6:tag*7]
    test_data=data[tag*7:]
    train_tokenized=[]
    # 分词
    train_tokenized=[words for words, _ in train_data]
    eval_tokenized=[words for words, _ in train_data]
    test_tokenized=[words for words, _ in train_data]
    print(train_tokenized[0:3])
    # 训练集词组
    vocab=set(chain(*train_tokenized))
    vocab_size=len(vocab)
    
    word_to_idx  = {word: i+1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0
    idx_to_word = {i+1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = '<unk>'
    # 
    train_features = torch.tensor(pad_samples(encode_samples(train_tokenized, vocab)))
    train_labels = torch.tensor([score for _, score in train_data])
    eval_features = torch.tensor(pad_samples(encode_samples(eval_tokenized, vocab)))
    eval_labels = torch.tensor([score for _, score in eval_data])
    test_features = torch.tensor(pad_samples(encode_samples(test_tokenized, vocab)))
    test_labels = torch.tensor([score for _, score in test_data])
    return train_data, eval_data, test_data, train_tokenized, eval_tokenized, vocab, vocab_size, word_to_idx, idx_to_word
if __name__=="main":
    train_data, eval_data, test_data, train_tokenized, eval_tokenized, vocab, vocab_size, word_to_idx, idx_to_word=get_data()
    # wvmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',
    #                                                        binary=True, encoding='utf-8')