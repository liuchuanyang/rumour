from data import *
from model import *
import torch.optim as optim
word_to_idx={}
def encode_samples(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features

def pad_samples(features, maxlen=140, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features
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
    global word_to_idx
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
    # return train_data, eval_data, test_data, train_tokenized, eval_tokenized, vocab, vocab_size, word_to_idx, idx_to_word
    # 返回参数
    return train_data, eval_data, test_data, \
    train_tokenized, eval_tokenized, test_tokenized, \
    vocab, vocab_size, \
    word_to_idx, idx_to_word, \
    train_features, train_labels,eval_features, eval_labels, test_features, test_labels
if __name__=="__main__":
    train_data, eval_data, test_data, \
    train_tokenized, eval_tokenized, test_tokenized,\
    vocab, vocab_size, \
    word_to_idx, idx_to_word,\
    train_features, train_labels,eval_features, eval_labels, test_features, test_labels =get_data()
    print(test_features)
    wvmodel = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin',
                                                           binary=True, encoding='utf-8')
    # 训练
    num_epochs = 5
    embed_size = 300
    num_hiddens = 100
    num_layers = 2
    bidirectional = True
    batch_size = 64
    labels = 2
    lr = 0.8
    device = torch.device('cuda:0')
    weight = torch.zeros(vocab_size+1, 300)

    for i in range(len(wvmodel.index2word)):
        try:
            index = word_to_idx[wvmodel.index2word[i]]
        except:
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(
            idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
    use_gpu = True      
    net = SentimentNet(vocab_size=(vocab_size+1), embed_size=embed_size,
                   num_hiddens=num_hiddens, num_layers=num_layers,
                   bidirectional=bidirectional, weight=weight,
                   labels=labels, use_gpu=use_gpu)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    test_set = torch.utils.data.TensorDataset(test_features, test_labels)

    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=False)
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, test_losses = 0, 0
        train_acc, test_acc = 0, 0
        n, m = 0, 0
        for feature, label in train_iter:
            n += 1
            net.zero_grad()
            feature = Variable(feature.cuda())
            label = Variable(label.cuda())
            score = net(feature)
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            train_acc += accuracy_score(torch.argmax(score.cpu().data,
                                                    dim=1), label.cpu())
            train_loss += loss
        with torch.no_grad():
            for test_feature, test_label in test_iter:
                m += 1
                test_feature = test_feature.cuda()
                test_label = test_label.cuda()
                test_score = net(test_feature)
                test_loss = loss_function(test_score, test_label)
                test_acc += accuracy_score(torch.argmax(test_score.cpu().data,
                                                        dim=1), test_label.cpu())
                test_losses += test_loss
        end = time.time()
        runtime = end - start
        print('epoch: %d, train loss: %.4f, train acc: %.2f, test loss: %.4f, test acc: %.2f, time: %.2f' %
            (epoch, train_loss.data / n, train_acc / n, test_losses.data / m, test_acc / m, runtime))
        sys.stdout.flush()