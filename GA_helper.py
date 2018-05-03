#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:52:14 2017

@author: xuweijia
"""
import torch
from torch.autograd import Variable
import numpy as np
from tqdm import trange

def to_var(np_input,use_cuda,evaluate=False):
    # if evaluate, volatile=True, no grad be computed
    if use_cuda:
        output=Variable(torch.from_numpy(np_input),volatile=evaluate).cuda()
    else:
        output=Variable(torch.from_numpy(np_input),volatile=evaluate)
    return output

def to_vars(np_inputs,use_cuda,evaluate=False):
    return [to_var(np_input,use_cuda,evaluate) for np_input in np_inputs]

def gru(rnn_model,batch_seq,batch_seq_mask):
# input:       B,T,D
# input_mask   B,T            (real length 1,1,1,0,0)
  sequence_length=torch.sum(batch_seq_mask,1).squeeze(-1)         # B,
#  print(sequence_length.type())
#  print(sequence_length.size())
  sort_len,sort_index=sequence_length.sort(dim=0,descending=True) # B,
#  print(sort_index.type())
#  print(sort_index.size())
# sorted input:B,T,D
  sorted_batchseq=batch_seq[sort_index.data]
# pack input[0]:seq_len,D    every unvoid word embeddding
# pack_input[1]:T,           every time stamp word number
  pack_seq=torch.nn.utils.rnn.pack_padded_sequence(sorted_batchseq,sort_len.data.cpu().numpy(),batch_first=True)
  output_pack,hn=rnn_model(pack_seq)
  # unpack
  # output:   B,T,D
  output,out_seq_len=torch.nn.utils.rnn.pad_packed_sequence(output_pack,batch_first=True)
  # ori_order:B,T,D
  _,original_index=sort_index.sort(dim=0,descending=False)
  original_output=output[original_index.data]
  return original_output,sequence_length,hn # 2,B,h

def att_sum(t1,t2):
    # B,T,2h
    return t1 + t2

def att_mul(t1,t2):
    # B,T,2h
    return torch.mul(t1 , t2)

def att_cat(t1,t2):
    # B,T,2h
    return torch.cat([t1,t2],dim=-1) # B,T,4h

def feat_fuc(dw,qw):
    # dw:B,T
    # qw:B,Q
    feat=np.zeros(dw.shape)
    bsize=dw.shape[0]
    #print("feat batch %d, T:%d"%(bsize,dw.shape[1]))
#    # every batch's feature
#    #feat: B,T
#    if bsize==1:
#        feat=np.in1d(dw,qw)# (T,)
#    else:
    for i in range(bsize):
        feat[i,:]=np.in1d(dw[i,:],qw[i,:]) #(B,T)
    return feat.astype('int32')

def evaluate(model, data, use_cuda):
    acc = loss = n_examples = 0
    for dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos, fnames in data:
        
        bsize = dw.shape[0]
        feat=feat_fuc(dw,qw)
        
        dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,feat=to_vars(\
        [dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,feat],use_cuda,evaluate=True)
        
        loss_batch,acc_batch=model(dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,feat)
        
        loss+=loss_batch.cpu().data.numpy()[0]*bsize
        acc+=acc_batch.cpu().data.numpy()[0]
        n_examples += bsize
    # finish all ex in valid
    return loss/n_examples,acc/n_examples


#def evaluate(model, data, use_cuda):
#    acc = loss = n_examples = 0
#    tr = trange(
#        len(data),
#        desc="loss: {:.3f}, acc: {:.3f}".format(0.0, 0.0),
#        leave=False)
#    for dw, dw_m,qw,qw_m,dt,qt,tt,tm, \
#        answear, candidate, candi_m, cloze_pos, fnames in data:
##    for dw, dt, qw, qt, a, m_dw, m_qw, tt, \
##            tm, c, m_c, cl, fnames in data:
#        bsize = dw.shape[0]
#        n_examples += bsize
#        f=feat_fuc(dw,qw)
#        dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,f=to_vars(\
#        [dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,f], use_cuda=use_cuda,evaluate=True)
#        
##        loss_, acc_ = model(dw, dt, qw, qt, answear, dw_m, qw_m, tt,
##                                tm, candidate, candi_m, cloze_pos, fnames,f)
#        loss_, acc_ = model(dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,f)
#        
#        
#        _loss = loss_.cpu().data.numpy()[0]
#        _acc = acc_.cpu().data.numpy()[0]
#        loss += _loss
#        acc += _acc
#        tr.set_description("loss: {:.3f}, acc: {:.3f}".
#                           format(_loss, _acc / bsize))
#        tr.update()
#    tr.close()
#    return loss / len(data), acc / n_examples
