import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from GA_helper import gru,att_sum,att_mul,att_cat

class GA_reader(nn.Module):
    def __init__(self,gru_size,vocab_size,embedding_size,embedding_init,train_emb,use_char,char_hidden,n_chars,char_dim,n_layers,gating_fn,use_feat,dropout):
        super(GA_reader, self).__init__()
        self.n_layers =n_layers
        self.hidden_size=gru_size
        self.embedding_init=embedding_init
        self.vacab_size=vocab_size
        self.embed_size=embedding_size # 100
        self.embed=torch.nn.Embedding(vocab_size,embedding_size)
        self.gating_fn = gating_fn
        self.use_char=use_char
        self.use_feat=use_feat
        if embedding_init is not None :
            # turn to tensor
            self.embed.weight.data.copy_(torch.from_numpy(embedding_init))
            # embed.weight=nn.Parameter(embedding_init)
        if not train_emb:
            # weight:Varible
            self.embed.weight.requires_grad=False
        # use in doc once,in query n_layyers
        if use_char:
            self.use_char=use_char
            self.char_dim=char_dim #25 
            self.char_hidden=char_hidden # 50
            self.char_embed=torch.nn.Embedding(n_chars,char_dim)
            self.char_gru=nn.GRU(self.char_dim,self.char_hidden,batch_first=True,bidirectional=True,dropout=dropout) # get 2*batch*char_hidden
            self.char_forward=nn.Linear(self.char_hidden,self.char_hidden)
            self.char_backward=nn.Linear(self.char_hidden,self.char_hidden)
            
        self.dropout_layyer=nn.Dropout(dropout)
        
        self.doc_Modulelist=nn.ModuleList()
        self.qry_Modulelist=nn.ModuleList()
        if use_char:
            self.input_size=self.embed_size+self.char_hidden
        else:
            self.input_size=self.embed_size
            
        for i in range(n_layers-1):
            doc_gru=nn.GRU(self.input_size if i==0 else 2*self.hidden_size,self.hidden_size,batch_first=True,bidirectional=True,dropout=dropout)
            qry_gru=nn.GRU(self.input_size,                                self.hidden_size,batch_first=True,bidirectional=True,dropout=dropout)
            self.doc_Modulelist.append(doc_gru) # model.add_module(name,module) child module
            self.qry_Modulelist.append(qry_gru)
        # feat just use in doc to show if di in qry
        if self.use_feat:
            self.feat_embed=torch.nn.Embedding(2,2)
            self.final_x_size=2*self.hidden_size+2
        else:
            self.final_x_size=2*self.hidden_size
        #final layer
        self.final_doc_layer = nn.GRU(self.final_x_size,self.hidden_size,batch_first=True,bidirectional=True,dropout=dropout)
        self.final_qry_layer = nn.GRU(self.input_size,  self.hidden_size,batch_first=True,bidirectional=True,dropout=dropout)
        
    def forward(self,dw, dw_m,qw,qw_m,dt,qt,tt,tm, answear, candidate, candi_m, cloze_pos,feat):
        dw_embed=self.embed(dw.long()) # B,T,embed
        qw_embed=self.embed(qw.long())
        
        if self.use_char:
            t_embed=self.char_embed(tt.long()) # n,T,embeddding_size
            t_embed_gru,t_real_len,hn=gru(self.char_gru,t_embed,tm) # n,T,2h / n,
            hT_f=hn[0]#n,h
            hT_b=hn[1]#n,h
#            index=(t_real_len-1).view(-1,1).exapnd(t_embed_gru.size(0),t_embed_gru.size(-1)).unsqueeze(1)  # n,1,2h
#            out_T=t_embed_gru.gather(1,index).squeeze(1)  # n,2h
#            hT_f=hT[:,:self.char_hidden]                  # n,h
#            back_index=torch.zeros_like(index)            #n,1,2h
#            # forward h0// backward hT
#            out_0=t_embed_gru.gather(1,back_index).squeeze(1)  # n,2h
#            hT_b=hT[:,self.char_hidden:]                       # n,h
            t_forword =self.char_forward(hT_f)
            t_backward=self.char_forward(hT_b)
            # every token's embedding
            t_merge=t_forword+t_backward                        # n,h
            # dt'embed
            d_char_index=dt.long().view(-1)                     # B*T_doc,
            d_char_embed=t_merge.index_select(0,d_char_index)    # B*T_doc,h
            d_char_embed=d_char_embed.view(dt.size(0),-1,self.char_hidden) # B,T_doc,h
            dw_embed=torch.cat([dw_embed,d_char_embed],dim=-1)
            # qt'embed
            q_char_index=qt.long().view(-1)                     # B*T_qry,
            q_char_embed=t_merge.index_select(0,q_char_index)    # B*T_qry,h
            q_char_embed=q_char_embed.view(qt.size(0),-1,self.char_hidden) # B,T_qry,h
            qw_embed=torch.cat([qw_embed,q_char_embed],dim=-1)

        # ---first n-1 layers---
        for i in range(self.n_layers-1):
            doc_gru=self.doc_Modulelist[i]
            qur_gru=self.qry_Modulelist[i]
            
            doc_di_embed,_,_=gru(doc_gru,dw_embed,dw_m) # B,T,2h
            qur_qi_embed,_,_=gru(qur_gru,qw_embed,qw_m) # B,Q,2h
            qur_qi_embed_transpose=qur_qi_embed.permute(0,2,1)# B,2h,Q
            
            inter=torch.bmm(doc_di_embed,qur_qi_embed_transpose)  # B,T,Q
            b2q_w=F.softmax(inter.view(-1,inter.size(-1))).view_as(inter) # B,T,Q  softmax attention to Q
            q_mask=qw_m.unsqueeze(1).float().expand_as(b2q_w)                      # B,T,Q
            b2q_w=b2q_w*q_mask                                             # no attention to void qi
            b2q_w_norm=b2q_w / torch.sum(b2q_w,2).expand_as(b2q_w)         # normalized attention, to get att vector 
            #every di's attention vector B,T,Q      B,Q,2h
            weighted_attention=torch.bmm(b2q_w_norm,qur_qi_embed)     # B,T,2h
            new_xi=eval(self.gating_fn)(doc_di_embed, weighted_attention)      # B,T,2h (if not cat)
            dw_embed=self.dropout_layyer(new_xi)  # B,T,2h (xi before gru)
        # ---final layer---
        if self.use_feat:# dw in qry or not
            feat_emb=self.feat_embed(feat.long())               # B,T,2
            dw_embed=torch.cat([dw_embed,feat_emb],dim=-1)      # B,T,final_x_size  xi
        doc_embed,_,_=gru(self.final_doc_layer,dw_embed,dw_m)        # B,T,2h            di
        qry_embed,_,_=gru(self.final_qry_layer,qw_embed,qw_m)        # B,Q,2h            qi
        # ---@ph's embed---
        bsize=qry_embed.size()[0]
        cloze_pos_expand=cloze_pos.view(-1,1).expand(bsize,qry_embed.size(2)).unsqueeze(1) # B,1,2h
        cloze_embed=qry_embed.gather(1,cloze_pos_expand.long()).squeeze(1)                 # B,2h
        #---@ph's attention---
        s=torch.bmm(doc_embed,cloze_embed.unsqueeze(-1)).squeeze(-1) # B,T  to each word
        s_mask=F.softmax(s) * candi_m.float()                        # B,T  softmax attention to D /only word in candidate
        s_normal=s_mask/torch.sum(s_mask,dim=1).expand_as(s_mask)    # B,T  normalize
        # ---sum attention---
        cand_prob=torch.bmm(s_normal.unsqueeze(1),candidate.float()).squeeze(1)   # B,N_cand    row:each candi's probality(include all words in it)
        # ---compute loss,acc---
        # CE loss
        answear_index=answear.unsqueeze(1) # B,1
        predict_prob=cand_prob.gather(1,answear_index.long()) # B,1
        loss=torch.mean(-torch.log(predict_prob))
        # Accuracy
        _,predict_cand=torch.max(cand_prob,1)  # B,1
        Accuracy=torch.sum(torch.eq(predict_cand.view(-1).float(),answear_index.view(-1).float()))
        return loss,Accuracy
            