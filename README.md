Gated Attention Reader Model
========================

This is a pytorch implemention of Gated-attention Reader Model([Gated-Attention Readers for Text](https://arxiv.org/abs/1606.01549))

Prerequisites
========================

* Python 3.6
* Pytorch 0.1.12_2

Data
=================
You can find data and pretained word embeddings file though this [link](https://drive.google.com/drive/folders/0B7aCzQIaRTDUZS1EWlRKMmt3OXM).  

You need to create your own data file, put data and pretrained word embedding file in it.


Training
=================
train dailymail data from raw:    
python main.py --data_dir 'path to your data file'  --dataset dailymail   --embed_dir word2vec_glove.txt
