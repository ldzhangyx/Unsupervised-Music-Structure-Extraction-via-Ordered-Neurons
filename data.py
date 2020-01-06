import os
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, extend = False):
        if extend:
            self.dictionary = Dictionary()
            self.train = self.tokenize_extend(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize_extend(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize_extend(os.path.join(path, 'test.txt'))
        else:
            self.dictionary = Dictionary()
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def tokenize_extend(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = list()
            token = 0
            for line in f:
                words = ['<eos>'] + line.split() + ['<eos>']
                ids.append(torch.Tensor([self.dictionary.word2idx[word] for word in words])) # 分句
                token += 1

        return ids

# import random
def data_spliter(data_path, output_path):
    split_prec = [0, 0.8, 0.9, 1]
    with open(data_path, 'r') as file:
        data_list = file.readlines()
    # random.shuffle(data_list)
    split_prec = [int(i * len(data_list)) for i in split_prec]
    file_list = ['train.txt','valid.txt','test.txt']
    for i in range(3):
        path = '/'.join([output_path, file_list[i]])
        with open(path, 'w') as output:
            output.writelines(data_list[split_prec[i]:split_prec[i+1]])
#

# for i in ['A','B','C']:
#     for j in ['minor','major']:
#         data_path = "/gpfsnyu/home/yz6492/on-lstm/data/hooktheory/" + i + "/{}_list.txt".format(j)
#         output_path = "/gpfsnyu/home/yz6492/on-lstm/data/hooktheory/" + i + "/" + j + "/list/"
#         data_spliter(data_path, output_path)
#         print(output_path)

# for i in ['A','B','C']:
#     for j in ['minor','major']:
#         data_path = "/gpfsnyu/home/yz6492/on-lstm/data/hooktheory/{}/{}.txt".format(i,j)
#         output_path = "/gpfsnyu/home/yz6492/on-lstm/data/hooktheory/{}/{}/".format(i,j)
#         data_spliter(data_path, output_path)
#         print(output_path)

#
# for i in ['A']:
#     data_path = "/gpfsnyu/home/yz6492/on-lstm/data/billboard/" + i + "/chord_list.txt"
#     output_path = "/gpfsnyu/home/yz6492/on-lstm/data/billboard/" + i + "/list/"
#     data_spliter(data_path, output_path)
#     print(output_path)
#
# for i in ['A']:
#     data_path = "/gpfsnyu/home/yz6492/on-lstm/data/billboard/{}/chord.txt".format(i)
#     output_path = "/gpfsnyu/home/yz6492/on-lstm/data/billboard/{}/".format(i)
#     data_spliter(data_path, output_path)
#     print(output_path)

# for i in ['4bar']:
#     for j in ['minor','major']:
#         data_path = "/gpfsnyu/home/yz6492/on-lstm/data/hooktheory_melody/{}/{}.txt".format(i,j)
#         output_path = "/gpfsnyu/home/yz6492/on-lstm/data/hooktheory_melody/{}/{}/".format(i,j)
#         data_spliter(data_path, output_path)
#         print(output_path)
#
# for i in ['4bar']:
#     for j in ['minor','major']:
#         data_path = "/gpfsnyu/home/yz6492/on-lstm/data/hooktheory_melody/" + i + "/{}_list.txt".format(j)
#         output_path = "/gpfsnyu/home/yz6492/on-lstm/data/hooktheory_melody/" + i + "/" + j + "/list/"
#         data_spliter(data_path, output_path)
#         print(output_path)

# data_path = "/gpfsnyu/home/yz6492/on-lstm/data/gttm/gttm_list.txt"
# output_path = "/gpfsnyu/home/yz6492/on-lstm/data/gttm/gttm/list/"
# data_spliter(data_path, output_path)
# print(output_path)