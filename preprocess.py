#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:00:10 2017

self.data_process

@author: xuweijia
"""

import os   # operate path
import glob # get document name. glob.glob(path)
import numpy as np
MAX_WORD_LEN=10
SYMB_S="@begin"
SYMB_E="@end"
data_dir='/media/xuweijia/00023F0D000406A9/fake_data/wdw'

class data_holder:
    def __init__(self, dictionary, training, validation, test, n_entities):
        self.dictionary = dictionary
        self.training =np.asarray(training)
        self.validation = np.asarray(validation)
        self.test = np.asarray(test)
        self.vocab_size = len(dictionary[0])
        self.n_chars = len(dictionary[1])
        self.n_entities = n_entities
        self.inv_dictionary = {v: k for k, v in dictionary[0].items()}

class preprocessing():
    def preprocess(self,data_dir,use_chars=True,no_training_set=False):
        # vocab file should store in each dataset;
        # note:just join, not create
        vocab_file=os.path.join(data_dir,'vocab.txt')
        word_dict,char_dict,num_entities=self.make_dictionary(data_dir,vocab_file)
        # training
        if no_training_set:
            training = None
        else:
            print ("preparing training data ...")
            training_path=os.path.join(data_dir,'training')
            training=self.parse_all_file(training_path,word_dict,char_dict, use_chars)
        print ("preparing validation data ...")
        validation_path=os.path.join(data_dir,'validation')
        validation=self.parse_all_file(validation_path,word_dict,char_dict, use_chars)
        print ("preparing test data ...")
        test_path=os.path.join(data_dir,'test')
        test=self.parse_all_file(test_path,word_dict,char_dict, use_chars)
        
        dictionary=(word_dict,char_dict)
        data=data_holder(dictionary, training, validation, test, num_entities)
        return data
    
    # make dict / words write in vocab_file
    def make_dictionary(self,data_dir,vocab_file):
        if os.path.exists(vocab_file):
            print('vocab file found, create vocab list...')
            # f.readlines().strip()
            vocabulary=list(map(lambda x:x.strip(), open(vocab_file).readlines()))
        else:
            vocab_set=set()
            print('no vocab file exist,create from dataset...')
            # get all example files' name list
            file_names=[] 
            file_names+=glob.glob(os.path.join(data_dir,'test/*.question'))
            file_names+=glob.glob(os.path.join(data_dir,'validation/*.question'))
            file_names+=glob.glob(os.path.join(data_dir,'training/*.question'))
            n=0.
            for file in file_names:
                fp=open(file)
                
                fp.readline()
                fp.readline()
                # doc
                # document=fp.readline().strip().split()
                document=fp.readline().split()
                fp.readline()
                # query
                #query=fp.readline().strip().split()
                query=fp.readline().split()
                fp.close()
                
                vocab_set|=set(document)|set(query)
                # show progress
                n+=1
                if n%10000==0:
                    print ('%3f %%'%(n/len(file_names)*100))
            
            entities= set([v for v in vocab_set if v.startswith('@entity')])
            # token include @placehoder, @begin and @end   
            tokens= vocab_set.difference(entities)
            tokens.add(SYMB_S)
            tokens.add(SYMB_E)
            
            vocabulary=list(tokens)+list(entities)
            #vocabulary.insert(0, UNK) # any word's not in doc+quy,index 0
        
            # write vocubulary in file,store in dataset folder
            # seperate by '\n'
            f=open(vocab_file,'w')
            f.write('\n'.join(vocabulary))
            f.close()
            
       # word dictionary: word -> index
        vocab_size=len(vocabulary)
        word_dict=dict(zip(vocabulary,range(vocab_size)))
        
        # char dictionary: char -> index
        char_set=set([c for w in vocabulary for c in list(w)])
        char_set.add(' ')
        char_list=list(char_set)
        char_dict=dict(zip(char_list,range(len(char_set))))
        
        num_entities = len(
        [v for v in vocabulary if v.startswith('@entity')])
            
        print ("vocab_size = %d" % vocab_size)
        print ("num characters = %d" % len(char_set))
        print ("%d anonymoused entities" % num_entities)
        print ("%d other tokens (including @placeholder, %s and %s)" % (
                vocab_size-num_entities, SYMB_S, SYMB_E))
        
        return word_dict,char_dict,num_entities

    def parse_one_file(self,file,word_dict,char_dict, use_chars):
        with open(file) as f:
            fp=f.readlines()
        #document_raw=fp[2].strip().split()
        #query_raw=fp[4].strip().split()
        #answer_raw=fp[6].strip().split()
        document_raw=fp[2].split()
        query_raw=fp[4].split()
        answer_raw=fp[6].strip()
        candidates_raw=list(map(lambda x: x.strip().split(':')[0].split() ,fp[8:])) # candidate answers,just @entityid,maybe more than 1 word
        
        # add SOS,EOS for query
        query_raw.insert(0, SYMB_S)
        query_raw.append(SYMB_E)
        
        # cloze's position in query
        try:
            cloze = query_raw.index('@placeholder')  # cloze place in qury
        except ValueError:
            print ('@placeholder not found in ', file, '. Fixing...')
            at = query_raw.index('@')  # where @ is
            query_raw = query_raw[:at] + [''.join(query_raw[at:at+2])] + query_raw[at+2:]
            # new query @--> @placeholder
            cloze = query_raw.index('@placeholder')
        
        # tokens/entities --> indexes
        doc_words=list(map(lambda w:word_dict[w],document_raw)) 
        qury_words=list(map(lambda w:word_dict[w],query_raw))
        
        # any word in answear/candidate  not in vocab,defualt index 0 may be many words in an answear
        answer_words=list(map(lambda w:word_dict.get(w,0),answer_raw.split()))

        cand_words=[list(map(lambda w:word_dict.get(w,0) ,c)) for c in candidates_raw] # every candidate
        
        # use char:C(w): char's index embedding
        # not emerge:    take value of dict[' ']
        # return char list of every word
        char_2_index=lambda c:char_dict.get(c,char_dict[' '])
        charlist_2_index=lambda w: list(map(char_2_index, list(w)[:MAX_WORD_LEN]))
        if use_chars:
            doc_chars =list(map(charlist_2_index,document_raw))
            qry_chars =list(map(charlist_2_index, query_raw))
        else:
            doc_chars, qry_chars = [], []

        return doc_words, qury_words, answer_words, cand_words, doc_chars, qry_chars, cloze
    
    def parse_all_file(self,files_path,word_dict,char_dict, use_chars):
        files=glob.glob(files_path+'/*.question')
        # 每个训练样本，后面跟文件名
        questions = [self.parse_one_file(f,word_dict,char_dict, use_chars) + (f,) for f in files]
        return questions
    # write all sample's  ducument ,' ',query into a  text_file.txt  
    def gen_text_for_word2vec(self, data_dir, text_file):

            fnames = []
            fnames += glob.glob(data_dir + "/training/*.question")

            out = open(text_file, "w")

            for fname in fnames:
                
                fp = open(fname)
                fp.readline()
                fp.readline()
                document = fp.readline()
                fp.readline()
                query = fp.readline()
                fp.close()
                
                out.write(document.strip())
                out.write(" ")
                out.write(query.strip())

            out.close()
        
#pre=preprocessing()
#data_hold=pre.preprocess(data_dir,use_chars=True,no_training_set=False)
#train,valid,test,dic,n_chars=data_hold.training,data_hold.validation,data_hold.test,data_hold.dictionary,data_hold.n_chars
## docu,query,ans,cani,docu)char,query_char,
## return       
#pre.gen_text_for_word2vec(data_dir,os.path.join(data_dir,'cbtcn_D_Q_train.txt'))
