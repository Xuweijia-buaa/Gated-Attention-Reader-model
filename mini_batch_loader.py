# -*- coding: utf-8 -*-
"""
create a class to iteratively load mini_batch
"""
#from config import  MAX_WORD_LEN
MAX_WORD_LEN=10
import numpy as np
import random
# samples:doc_words, qury_words, answer_words, cand_words, doc_chars, qry_chars, cloze,file_name

class mini_batch_loader:
    # samples: all the index of  
    def __init__(self,samples,batch_size,sample_rate=1.0,shuffle=True,query_len_all=False,len_bin=True):
        if sample_rate == 1.0:
            self.samples = samples
        else:
            self.samples = random.sample(
                samples, int(sample_rate * len(samples)))
        self.batch_size=batch_size
        # max query length in all sample/ or in a batch
        self.max_qury_len= max(list(map(lambda x: len(x[1]),self.samples)))
        # print('max_query'%self.max_qury_len)
        # max candidate in all samples
        self.max_num_cand = max(list(map(lambda x:len(x[3]), self.samples)))
        self.max_word_len = MAX_WORD_LEN
        # test/valid false
        self.shuffle=shuffle
        self.query_len=query_len_all
        self.len_bin=len_bin
        if self.len_bin:
            # dict len:sample index
            self.bins = self.build_bins(self.samples)
            self.reset_bin()
        else:
            self.reset()
    
    # make object iterable
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.batch_pool)
    
    
    def build_bins(self,questions):
        """
        returns a dictionary
            key: document length (rounded to the powers of two)
            value: indexes of questions with document length equal to key
        """
        # round the input to the nearest power of two
        #  2^log(x-1) +1
        round_to_power = lambda x: 2**(int(np.log2(x-1))+1)
        #   每个文件中，ducument words的数目，约到最近的2的指数，
        doc_len = map(lambda x:round_to_power(len(x[0])), questions)
        bins = {}
        for i, l in enumerate(doc_len):
            if l not in bins:
                bins[l] = []
            bins[l].append(i)
        return bins

    def reset_bin(self):
        """new iteration"""
        self.ptr= 0

        # randomly shuffle the question indices in each bin
        if self.shuffle:
            for ixs in self.bins.values():
                random.shuffle(ixs)

        # construct a list of mini-batches where each batch is a list of question indices
        # questions within the same batch have identical max document length 
        self.batch_pool = []
        for l, ixs in self.bins.items():
            n = len(ixs)
            k = n//self.batch_size if n % self.batch_size == 0 else n//self.batch_size+1
            ixs_list = [(ixs[self.batch_size*i:min(n, self.batch_size*(i+1))],l) for i in range(k)]
            self.batch_pool += ixs_list

        # randomly shuffle the mini-batches
        if self.shuffle:
            random.shuffle(self.batch_pool)
    
    def reset(self):
        self.it=0
        n=len(self.samples)
        # disordered example id
        index=np.arange(n)
        if self.shuffle:
            np.random.shuffle(index)
            # sample must be nparray type
            self.samples=self.samples[index] 
        # store batch example's index
        batch_pool=[]    
        batch_start_index=np.arange(0,n,self.batch_size)        

        for index in range(len(batch_start_index)):
            batch=np.arange(batch_start_index[index],min(batch_start_index[index]+self.batch_size,n))
            batch_pool.append(batch)
        
        self.batch_pool=batch_pool
        
    def __next__(self):
        if not self.len_bin:
            # do not change length into 2^
            if self.it==len(self.batch_pool):
                # one epoch finished
                self.reset()
                raise StopIteration
            # this batch's sample index 0-5,6-10    
            ixs=self.batch_pool[self.it]
            current_batch_size=len(ixs)
            current_doc_max= np.max(list(map(lambda x:len(x[0]),[self.samples[i] for i in ixs])))
            current_qur_max= np.max(list(map(lambda x:len(x[1]),[self.samples[i] for i in ixs])))
        else:
            if self.ptr == len(self.batch_pool):
               self.reset_bin()
               raise StopIteration()
            ixs = self.batch_pool[self.ptr][0]
            current_batch_size = len(ixs)
#            current_doc_max = self.batch_pool[self.ptr][1] #2^n
#            print(current_doc_max)
            
            current_doc_max= np.max(list(map(lambda x:len(x[0]),[self.samples[i] for i in ixs])))# real doc
#            print(current_doc_max)
            
            current_qur_max=np.max(list(map(lambda x:len(x[1]),[self.samples[i] for i in ixs]))) # real qury
#            print(current_qur_max)
            
              
        
        # create batch word index
        dw=np.zeros((current_batch_size,current_doc_max),dtype='int32')
        qw=np.zeros((current_batch_size,current_qur_max),dtype='int32')
        
        # mask,reflect real length
        dw_m=np.zeros((current_batch_size,current_doc_max),dtype='int32')
        qw_m=np.zeros((current_batch_size,current_qur_max),dtype='int32')
        
        # answear
        answear=np.zeros(current_batch_size,dtype='int32')
        
        #candidate:correspond to every candidate's position in doc
        # each sample's each candidate in doc
        candidate=np.zeros((current_batch_size,current_doc_max,self.max_num_cand),dtype='int16')
        # each sample's all candidates in doc
        candi_m=np.zeros((current_batch_size,current_doc_max),dtype='int32')
        
        # cloze position
        cloze_pos=np.zeros((current_batch_size,),dtype='int32')
        # sample file name
        fnames = ['']*current_batch_size
        # token(in this batch):token place 
        token_type={}
        for ex_id,ix in enumerate(ixs):
            # doc_words, qury_words, answer_words, cand_words, doc_chars, qry_chars, cloze
            doc_words, qury_words, answear_words, cand_words, doc_chars, qry_chars, cloze,file_name=self.samples[ix]
            
            dw[ex_id,:len(doc_words)]=np.array(doc_words)
            qw[ex_id,:len(qury_words)]=np.array(qury_words)
            
            dw_m[ex_id,:len(doc_words)]=1
            qw_m[ex_id,:len(qury_words)]=1
            
            for cand_index,cand_word in enumerate(cand_words):
                # maybe more than 1 word in a cand
                # all position in doc if emerge in cand
                pos=[ii for ii in range(len(doc_words)) if doc_words[ii] in cand_word]
                candidate[ex_id,pos,cand_index]=1
                candi_m[ex_id,pos]=1
                if answear_words==cand_word:
                    # anwear record candidate id
                    answear[ex_id]=cand_index
            
            cloze_pos[ex_id]=cloze
            fnames[ex_id]=file_name
            # dealing with tokens
            for t_pos,char_list in enumerate(doc_chars):
                token=tuple(char_list)
                if token not in token_type:
                    token_type[token]=[]
                # token show in 
                # ex_id example's doc's t_pos position in this batch
                token_type[token].append((0,ex_id,t_pos))
            for t_pos,char_list in enumerate(qry_chars):
                token=tuple(char_list)
                if token not in token_type:
                    token_type[token]=[]
                # token show in 
                # ex_id, qry_pos
                token_type[token].append((1,ex_id,t_pos))
        
        # correspond to token(char_list) id
        dt=np.zeros((current_batch_size,current_doc_max),dtype='int32')
        qt=np.zeros((current_batch_size,current_qur_max),dtype='int32')
        # tt each row: one token list
        tt=np.zeros((len(token_type),self.max_word_len),dtype='int32')
        # tm:char real length
        tm=np.zeros((len(token_type),self.max_word_len),dtype='int32')
        token_id=0
        # dt
        for token,position_list in token_type.items():
            # char_list
            tt[token_id,:len(token)]=list(token)
            tm[token_id,:len(token)]=1
            # replace dc,qc with char_token_index
            for (symbol,exid,pos) in position_list:
                if symbol==0:
                    dt[exid,pos]=token_id
                else:
                    qt[exid,pos]=token_id
            token_id+=1
        # next batch    
        if self.len_bin:
            self.ptr+=1
        else:
            self.it+=1
        return dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos, fnames

#train_loader=mini_batch_loader(train,10,shuffle=False,len_bin=True)
#valid_loader=mini_batch_loader(valid,10,shuffle=False,len_bin=True)
#test_loader=mini_batch_loader(test,10,shuffle=False,len_bin=True)        
#for dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos, fnames in test_loader:
#    dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos, fnames=dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos, fnames
    
