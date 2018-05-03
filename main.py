#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import argparse
import torch
import numpy as np
import time
import os
import logging
import shutil
from torch.nn.utils import clip_grad_norm

from preprocess import preprocessing
from mini_batch_loader import mini_batch_loader

from GA_reader import GA_reader
from GA_helper import feat_fuc,evaluate,to_vars
from load_embedding import load_word2vec_embeddings

from config import *

USE_CUDA = torch.cuda.is_available()


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser(
        description='Gated Attention Reader for \
        Text Comprehension Using PyTorch')
    parser.register('type', 'bool', str2bool)

    parser.add_argument('--data_dir',  dest='data_dir', type=str, default='data',  help='.../data')
    parser.add_argument('--dataset',   dest='dataset', type=str, default='wdw',    help='cnn || dailymail || cbtcn || cbtne || wdw')
    parser.add_argument('--embed_dir', dest='embed_file', type=str, default=None,  help='word2vec_glove.txt in data file')
    parser.add_argument('--seed', dest='seed', type=int, default=3,                help='Seed for different experiments with same settings')
    parser.add_argument('--run_mode',  dest='mode', type=int, default=0,           help='0-train+test, 1-test only, 2-val only')
    parser.add_argument('--nlayers',   dest='nlayers', type=int, default=3,        help='Number of reader layers')
    parser.add_argument('--use_default',dest='use_default_hp',type=str2bool,default=True,help=' from  config/commad line')
    
    ## hyper_parameter
    parser.add_argument('--gating_fn',dest='gating_fn', type=str, default='att_mul' , help='gating function:att_cat, att_sum, att_mul (default)')
    parser.add_argument('--gru_size', dest='nhidden',type=int, default=256,           help='size of word GRU hidden state')
    parser.add_argument('--char_gru_size', dest='char_nhidden',type=int, default=50,  help='size of char GRU hidden state')
    parser.add_argument('--use_feat', dest='use_feat',type=str2bool, default=False,   help='whether to use extra features')
    parser.add_argument('--train_emb',dest='train_emb',type=str2bool, default=True,   help='whether to train embed')
    parser.add_argument('--char_dim', dest='char_dim',type=int, default=0,            help='size of character GRU hidden state')
    parser.add_argument('--drop_out', dest='dropout' ,type=float, default=0.1,        help='dropout rate')
    parser.add_argument('--use_bin',  dest='use_bin' ,type=str2bool, default=True,    help='if construct 2^n backet and build batch')
    parser.add_argument('--plot_loss',dest='plot_loss' ,type=str2bool, default=True, help='if construct 2^n backet and build batch')
    args = parser.parse_args()
    print ("use default congifure is %d"%int(args.use_default_hp))
    hp_dict=vars(args)
#  use default config
    if args.use_default_hp:
        Defualt_hp_params=get_params(args.dataset)
        hp_dict.update(Defualt_hp_params)
    data_dir=os.path.join(hp_dict['data_dir'],hp_dict['dataset'])
    # data/'word_vector.txt'
    if hp_dict['embed_file']:
        embed_file=os.path.join(hp_dict['data_dir'],hp_dict['embed_file'])
        hp_dict['embed_file']=embed_file
        
    word_file_name=hp_dict['embed_file'].split('/')[-1].split('.')[0] if hp_dict['embed_file'] else 'None'
    save_path = ('experiments/'+hp_dict['dataset']+
        '_nhid%d'%hp_dict['nhidden']+'_nlayers%d'%hp_dict['nlayers']+
        '_dropout%.1f'%hp_dict['dropout']+'_%s'%word_file_name+'_chardim%d'%hp_dict['char_dim']+'_char_h %d'%hp_dict['char_nhidden']+
        '_train%d'%hp_dict['train_emb']+
        '_seed%d'%hp_dict['seed']+'_use-feat%d'%hp_dict['use_feat']+
        '_gf%s'%hp_dict['gating_fn'])
    if not os.path.exists(save_path): os.makedirs(save_path)
    return args,hp_dict,data_dir,save_path


def train(hp_dict,args,data_dir,save_path):
    use_chars=hp_dict['char_dim']>0
    # load data
    dp = preprocessing()
    data = dp.preprocess(
        data_dir,
        no_training_set=False,
        use_chars=use_chars
    )

    # build minibatch loader
    train_batch_loader = mini_batch_loader(
        data.training, BATCH_SIZE, sample_rate=1.0,len_bin=hp_dict['use_bin'])
    valid_batch_loader = mini_batch_loader(
        data.validation, BATCH_SIZE, shuffle=False,len_bin=hp_dict['use_bin'])
    test_batch_loader = mini_batch_loader(
        data.test, BATCH_SIZE, shuffle=False,len_bin=hp_dict['use_bin'])

    logging.info("loading word2vec file ...")
    embed_init, embed_dim = \
        load_word2vec_embeddings(data.dictionary[0], hp_dict['embed_file'],EMBED_SIZE)
    logging.info("embedding dim: {}".format(embed_dim))
    logging.info("initialize model ...")
    
    
    model=GA_reader(hp_dict['nhidden'],data.vocab_size,embed_dim,embed_init,hp_dict['train_emb'],
                    use_chars,hp_dict['char_nhidden'],data.n_chars,hp_dict['char_dim'],
                    hp_dict['nlayers'],hp_dict['gating_fn'],hp_dict['use_feat'],hp_dict['dropout'])
    
    if USE_CUDA:
        model.cuda()
    logging.info("Running on cuda: {}".format(USE_CUDA))
    # training phase
    opt = torch.optim.Adam(
        params=filter(
            lambda p: p.requires_grad, model.parameters()
        ),
        lr=LEARNING_RATE)
    
    shutil.copyfile('config.py',os.path.join(save_path,'config.py'))
#    
    # load existing best model
    if os.path.isfile(os.path.join(save_path,'best_model.pkl')):
        print('loading previously best model')
        model.load_state_dict(torch.load(os.path.join(save_path,'best_model.pkl')))
     # load existing train_model 
    elif os.path.isfile(os.path.join(save_path,'init_model.pkl')):
        print('loading init model')
        model.load_state_dict(torch.load(os.path.join(save_path,'init_model.pkl')))

        
    logging.info('-' * 50)
    logging.info("Start training ...")
    best_valid_acc = best_test_acc = 0
    for epoch in range(NUM_EPOCHS):
        new_max=False
        if epoch >= 2:
            for param_group in opt.param_groups:
                param_group['lr'] /= 2
        model.train()
        acc = loss = n_examples = it = 0
        start = time.time()

        for dw, dw_m,qw,qw_m,dt,qt,tt,tm, \
                 answear, candidate, candi_m, cloze_pos, fnames in train_batch_loader:
            n_examples += dw.shape[0]
            feat= feat_fuc(dw,qw)
#-------train-------#
            dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,feat=to_vars(\
           [dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,feat], use_cuda=USE_CUDA)
            
            loss_, acc_ = model(dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,feat) # tensor.float size 1
            #print(acc_.cpu().data.numpy())
            loss += loss_.cpu().data.numpy()[0] # numpy [1]
            acc += acc_.cpu().data.numpy()[0]
            it += 1
            opt.zero_grad()
            loss_.backward()
            clip_grad_norm(
                parameters=filter(
                    lambda p: p.requires_grad, model.parameters()
                ),
                max_norm=GRAD_CLIP)
            opt.step()
            if it % print_every == 0 \
                    or it % len(train_batch_loader) == 0:
                spend = (time.time() - start) / 60
                statement = "Epoch: {}, it: {} (max: {}), "\
                    .format(epoch, it, len(train_batch_loader))
                statement += "loss: {:.3f}, acc: {:.3f}, time: {:.1f}(m)"\
                    .format(loss / print_every, acc / n_examples, spend)
                logging.info(statement)
                del acc,loss,n_examples
                acc = loss = n_examples = 0
                start = time.time()
		# save every print
                torch.save(model.state_dict(), os.path.join(save_path,'init_model.pkl'))
                # torch.save(model,os.path.join(save_path,'init_model.pkl'))
#-------valid-------#
            if it % eval_every == 0:
                start = time.time()
                model.eval()
                test_loss, test_acc = evaluate(
                    model, valid_batch_loader, USE_CUDA)
                spend = (time.time() - start) / 60
                statement = "Valid loss: {:.3f}, acc: {:.3f}, time: {:.1f}(m)"\
                    .format(test_loss, test_acc, spend)
                logging.info(statement)
                if best_valid_acc < test_acc:
                    best_valid_acc = test_acc
                    new_max=True
                    # store best valid model
                    torch.save(model.state_dict(),os.path.join(save_path,'best_model.pkl'))
                    #torch.save(model,os.path.join(save_path,'best_model.pkl'))
                logging.info("Best valid acc: {:.3f}".format(best_valid_acc))
                model.train()
                start = time.time()
#-------test-------#
        start = time.time()
        model.eval()
        test_loss, test_acc = evaluate(
            model, test_batch_loader, USE_CUDA)
        spend = (time.time() - start) / 60
        logging.info("Test loss: {:.3f}, acc: {:.3f}, time: {:.1f}(m)"\
                     .format(test_loss, test_acc, spend))
        if best_test_acc < test_acc:
            best_test_acc = test_acc
        logging.info("Best test acc: {:.3f}".format(best_test_acc))
#        if not new_max: # until epoch no new accuracy
#            break


if __name__ == "__main__":
    args,hp_dict,data_dir,save_path= get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')
    logging.info(args)
    train(hp_dict,args,data_dir,save_path)
