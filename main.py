import codecs
from models import *
from data_helper import load_text_target_label, load_w2v
import os
import pickle
import numpy as np
import pandas as pd

import argparse

#inaam
def load_sentences(k): # for reading a file line by line
    valid_lines=[]
    with open(k) as word_file: # take a file
        valid_lines = word_file.read().split('\n') # split it line by line and then make a array of it
    return valid_lines # output the array of lines in the file

#inaam
def load_lines(k): # for reading a file line by line
    valid_lines=[]
    with open(k) as word_file: # take a file
        valid_lines = word_file.read().split('\n') # split it line by line and then make a array of it
    return valid_lines # output the array of lines in the file

def my_func(testfile):
    word2index = pickle.load(open(os.path.join('data', '14res', 'vocabulary.pkl'), "rb"))

    df = pd.read_csv("data/14res/test.tsv", sep="\t")
    #df.head()
    
    all_sentences = load_sentences(testfile)
    
    k=0
    for i in all_sentences:
        new_sentence = []
        i = i.split(" ")
        for j in range(0,len(i)):
            if i[j] != "" and i[j] != " " and i[j] != "  ":
                new_sentence.append(i[j])
        all_sentences[k] = new_sentence
        #print(all_sentences[k])
        k=k+1    
    
    k=0
    for i in all_sentences:
        all_sentences[k] = ' '.join(all_sentences[k])
        k=k+1
        
    for k in range(0,len(all_sentences)):
        all_sentences[k] = all_sentences[k].replace('.'," . ")
    
    a=0
    data2= df.head(0)
    for i in range(0,len(all_sentences)):
        my_arr = all_sentences[i]
        my_arr = my_arr.split(' ')
    
        my_new_arr = []
        for j in range(0,len(my_arr)):
            if my_arr[j] != "":
                try:
                    asdasd=word2index[my_arr[j]]
                    if j==0 :
                        my_new_arr.append(my_arr[j])
                    else:
                        my_new_arr.append(my_arr[j])
                except:
                    masdasd = 0
                    
        my_arr = my_new_arr
        my_arr_sentence = ' '.join(my_new_arr)
        
        target_words = load_lines('target_words.txt')
        
        my_new_arr = []
        for j in range(0,len(my_arr)):
            if target_words.count(my_arr[j])>0:
                my_new_arr.append(my_arr[j] + '\B')
            else:
                my_new_arr.append(my_arr[j] + '\O')
        
        if len(my_arr) != 0:            
            my_arr = ' '.join(my_new_arr)
            
            print(my_arr)
            data2 = data2.append(pd.Series([a,my_arr_sentence,my_arr,my_arr], index=data2.columns ), ignore_index=True)
        a=a+1
        
    data2.to_csv('myfile.tsv',sep="\t", mode='w+', columns=['s_id','sentence','target_tags','opinion_words_tags'], index=False)
    
    df = pd.read_csv("myfile.tsv", sep="\t")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_bs", type=int, default=8)

parser.add_argument("--EPOCHS", type=int, default=10) #default = 10 #inaam
parser.add_argument("--n_hidden", type=int, default=200)
parser.add_argument("--optimizer", type=str,  default="Adam")
parser.add_argument("--model", type=str, default="IOG")
parser.add_argument("--lr", type=float, default=0.2)
parser.add_argument("--freeze", type=bool, default=True)
parser.add_argument("--ds", type=str, default='14res')
parser.add_argument("--l1", type=int, default=1)
parser.add_argument("--l2", type=int, default=1)
parser.add_argument("--pos", type=bool, default=False)
parser.add_argument("--use_char", type=bool, default=False)
parser.add_argument("--use_crf", type=bool, default=False)
parser.add_argument("--projected", type=bool, default=False)
parser.add_argument("--use_elmo", type=bool, default=False)
parser.add_argument("--use_dev", type=bool, default=True)
parser.add_argument("--elmo_mode", type=int, default=6)
parser.add_argument("--elmo_mode2", type=int, default=0)
parser.add_argument("--attn_type", type=int, default=1)
parser.add_argument("--pos_size", type=int, default=30)
parser.add_argument("--char_embed_dim", type=int, default=30)
parser.add_argument("--test", type=int, default=0)
parser.add_argument("--test_model", type=str, default="TCD_LSTM_0.6327.pt")
args = parser.parse_args()

# torch.set_printoptions(profile="full")
print(args)

tag2id = {'B': 1, 'I': 2, 'O': 0}
# seed = 314159
# torch.manual_seed(seed)
# seed = torch.initial_seed()

def generate_mask(target):
        labels = numericalize_label(target, tag2id)
        index = np.nonzero(labels)
        #print(target)
        #print(index)
        start = index[0][0]
        end = index[0][-1]
        left_mask = np.asarray([1 for _ in range(len(target))])
        right_mask = np.asarray([1 for _ in range(len(target))])
        left_mask[end+1:] = 0
        right_mask[:start] = 0
        return left_mask, right_mask


def main():
    word2index = pickle.load(open(os.path.join('data', args.ds, 'vocabulary.pkl'), "rb"))
    init_embedding = np.load(os.path.join('data', args.ds, 'embedding_table.npy'))
    init_embedding = np.float32(init_embedding)

    # load train data
    print('loading train data...')
    train_text, train_target, train_label = load_text_target_label(os.path.join("data/", args.ds, 'train.tsv'))
    print(train_text[0])
    print(train_target[0])
    print(train_label[0])

    test_text, test_target, test_label = load_text_target_label(os.path.join("data/", args.ds, 'test.tsv'))

    model = NeuralTagger()
    model.train_from_data((train_text, train_target, train_label), (test_text, test_target, test_label), init_embedding, word2index, args)


def test():
    word2index = pickle.load(open(os.path.join('data', '14res', 'vocabulary.pkl'), "rb"))

    
    model_name = 'IOG_0.7598_0.8020.pt'
    my_func('test_file.txt')
    
    test_text, test_target, test_label = load_text_target_label('myfile.tsv')
    
    #print(test_text[0])
    #print(test_target[0])
    #print(test_label[0])
    
    test_raw_data = (test_text,test_target,test_label)
    
    TEXT = data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True,
                          include_lengths=True)
    LABEL_T = data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    LABEL_O = data.Field(sequential=True, use_vocab=False, pad_token=-1, batch_first=True)
    LEFT_MASK = data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    RIGHT_MASK = data.Field(sequential=True, use_vocab=False, pad_token=0, batch_first=True)
    
    model = NeuralTagger()
    
    rnn = torch.load("backup/%s" % model_name)
    rnn.eval() #inaam
    
    fields = [('text', TEXT), ('target', LABEL_T), ('label', LABEL_O), ('left_mask', LEFT_MASK), ('right_mask', RIGHT_MASK)]
    
    test_data = [
            [numericalize(text, word2index), numericalize_label(target, tag2id),
             numericalize_label(label, tag2id), *generate_mask(target)]
            for text, target, label in zip(*test_raw_data)]
    
    test_dataset = ToweDataset(fields, test_data)
    
    test_iter = data.Iterator(test_dataset, batch_size=len(test_text), shuffle=False, sort_within_batch=True,
                                  repeat=False,
                                  device=device if torch.cuda.is_available() else -1)
    
    for step, batch in enumerate(test_iter):
        ow = rnn(batch)   
        #print(len(ow))
    
    ow = ow.cpu().detach().numpy()    
    #print("here")
    #print(len(ow.shape))
    #print(ow)
    id2tag = {1: 'B', 2: 'I', 0: 'O'}
    results = []
    
    k=0
    for i in ow:
        res = np.argmax(i,axis=1)
        my_sent = test_text[k].split(" ")
        my_sent2 = []
        
        #print(len(my_sent))
        #print(len(res))
        #print(len(my_sent)==len(res))
        
        for m in range(0,len(my_sent)):
            my_sent2.append(str(my_sent[m])+"\\"+str(id2tag[res[m]]))
            
        results.append(" ".join(my_sent2))
        k=k+1
        
    my_res_file = open('results.txt',mode='w+')
    for i in range(0,len(results)):
        my_res_file.writelines(results[i]+"\n")
    my_res_file.close()
    print("\nResults have been written to results.txt")

if __name__ == '__main__':
    if args.test == 0:
        main()
    else:
        ow = test()
