# -*- coding: utf-8 -*-
from config import *
from config import get_params
import os
import random
import numpy as np

data_dir='data'
embedding_file='word2vec_glove.txt'


#def load_word2vec_embeddings(dictionary, embedding_file,EMBED_SIZE=100):
#    # return numpy array index correspond to dictionary
#    if embedding_file==None:
#        return None,EMBED_SIZE
#    else:
#        #embed_file=os.path.join(data_dir,embedding_file)  
#        fp= open(embedding_file, encoding='utf-8')           
#        _,embed_size=fp.readline().strip().split()
#        embed_size=int(embed_size)
#        
#        word_vector={}
#        for line in fp:
#            line_array=fp.readline().strip().split()
#            word_vector[line_array[0]]=\
#                                       np.array(list
#                                       (map(lambda x:(float(x)),line_array[1:])), dtype='float32')
#        fp.close()
#        
#        # make vocab_vector
#        V=len(dictionary)
#        W=np.random.randn(V,embed_size).astype('float32')
#
#        n=0 # record how many words in dic emerge in embed_file
#        for w,i in dictionary.items():
#            if w in word_vector:
#                W[i,:]=word_vector[w]
#                n+=1
#        print("{}/{} vocabs are initialized with word2vec embeddings."
#                 .format(n, V))
#        return W,embed_size
    
# W,embedding=load_word2vec_embeddings(dic[0],embedding_file,EMBED_SIZE)
        
def load_word2vec_embeddings(dictionary, vocab_embed_file,EMBED_DIM=100):
    if vocab_embed_file is None: return None, EMBED_DIM

    fp = open(vocab_embed_file)

    info = fp.readline().split()
    embed_dim = int(info[1])

    vocab_embed = {}
    for line in fp:
        line = line.split()
        vocab_embed[line[0]] = np.array(list(map(float, line[1:])), dtype='float32')
    fp.close()

    vocab_size = len(dictionary)
    W = np.random.randn(vocab_size, embed_dim).astype('float32')
    n = 0
    for w, i in dictionary.items():
        if w in vocab_embed:
            W[i,:] = vocab_embed[w]
            n += 1
    print("{}/{} vocabs are initialized with word2vec embeddings.".format(n, vocab_size))
#    print ("%d/%d vocabs are initialized with word2vec embeddings." % (n, vocab_size))
    return W, embed_dim
