import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.optim import Optimizer
import os
import spacy
import unicodedata
import re
nlp=spacy.load('en_core_web_md')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data=None
word2idx={}
idx2word={}
idx2vec={}
embed_filename='test.file'
word2vec_filename='dict.file'
data=[]
data_dir=os.path.abspath('.\\movie_lines.txt')
if os.name=='posix':
    data_dir.replace('\\','/')

'<10 padding >10 truncate'
LENGTH=10
TRAIN_LINE_NUM=500
ABBRS={'re':'are','m':'am','s':'is'}
SPECIAL=['BOS','EOS','OOV','PAD']

    
def normalizeString(s):
    s=s[:-3]
    s=s.lower()
    if '\n' in s:
        s.remove('\n')
    s=re.sub(r'([.!?])',r' \1',s)
    s=re.sub(r'[^a-zA-Z.!?]+',r' ',s)
    return s
    
    
def load_data():
    with open(data_dir,'rb') as f:
        count=0
        for line in f:
            sentence=str(line).split('+++$+++')[-1]
            sentence=normalizeString(sentence)
            sentence=sentence+' {0}'.format(SPECIAL[1])
            words=sentence.split()
            if len(words)<10:
                for i in range(10-len(words)):
                    words.append(SPECIAL[-1])
            elif len(words)>10:
                words=words[:10]
            data.append(words)
            #print(words)
            count+=1
            if count>=TRAIN_LINE_NUM:
                break


def embed_two_dim(idxtensor):
    'should use torch embeddings'
    vectensor=torch.Tensor(
        [[idx2vec[int(idx)] for idx in sentence] for sentence in idxtensor]
        ).to(device)
    return vectensor
    
    
'batch_first===False'
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,input,output,hidden_dim,loss_fn,max_len,batch_first=True):
        'input:input_num*sourceL*batch_size=1'
        super(EncoderDecoder,self).__init__() 
        self.encoder=encoder
        self.decoder=decoder
        self.batch_size=1
        self.input_num=input.size(0)
        self.enc_len=input.size(1)
        self.dec_len=output.size(1)
        print(input.size())
        print(output.size())
        self.hidden_dim=hidden_dim
        'parameter is changing, the data should be saved in the list'
        #self.encode_hid_list=[torch.rand(1,self.batch_size,hidden_dim)*0.01]
        self.dec_out_list=[]
        self.input=input
        self.output=output
        'NLLloss'
        self.loss_fn=loss_fn
        self.loss_val=0
        self.max_len=max_len
        
        self.BOS=torch.Tensor(idx2vec[word2idx[SPECIAL[0]]]).unsqueeze(0).unsqueeze(0).to(device)


    def forward(self,input_id):
        encode_hid=(0.01*torch.rand(1,self.batch_size,self.hidden_dim)).to(device)
        enc_input_seq=embed_two_dim(self.input[input_id])
        for enc_i in range(self.enc_len):
            enc_input=enc_input_seq[enc_i].unsqueeze(0)
            enc_output,encode_hid=self.encoder(enc_input,encode_hid)
        'omit c'
        output=self.BOS
        decode_hid=encode_hid
        dec_input_seq=embed_two_dim(self.output[input_id])
        for dec_i in range(self.dec_len):
            'teacher forcing'
            dec_input=dec_input_seq[dec_i].unsqueeze(0)
            output,decode_hid=self.decoder(dec_input,decode_hid)
            target=self.output[input_id,dec_i].to(device)
            self.loss_val+=self.loss_fn(output,target)


    def loss(self):
        self.loss_val=0
        for i in range(self.input_num):
            self.forward(i)
        return self.loss_val
    
            
    def predict(self,sentence):
        with torch.no_grad():
            inputidx=[[word2idx[word] for word in sentence]]
            enc_input_seq=embed_two_dim(inputidx)[0].unsqueeze(1)
            enc_len=len(sentence)
            encode_hid=(0.01*torch.rand(1,self.batch_size,self.hidden_dim)).to(device)
            for enc_i in range(enc_len):
                enc_input=enc_input_seq[enc_i].unsqueeze(0)
                enc_output,encode_hid=self.encoder(enc_input,encode_hid)
            decode_hid=encode_hid
            decoded_words=[]
            dec_input=self.BOS
            for dec_i in range(self.max_len):
                output,decode_hid=self.decoder(dec_input,decode_hid)
                _,wordidx=output[0].data.topk(1)
                idx=int(wordidx[0])
                dec_input=embed_two_dim([[idx]])
                decoded_words.append(idx2word[idx])
                if idx==word2idx[SPECIAL[1]]:
                    break
            return decoded_words
        
        
    def check_cuda_params(self):
        print(torch.cuda.memory_allocated())
        print('cuda contains:')
        print('input/output:{0}/{1}'.format(self.input.is_cuda,self.output.is_cuda))
        

class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Encoder,self).__init__()
        self.dim=input_size
        self.hidden_dim=hidden_size
        self.rnn=nn.GRU(input_size,hidden_size)


    def forward(self,x,hidden):
        output,hidden=self.rnn(x,hidden)
        return output,hidden


class Decoder(nn.Module):
    def __init__(self,hidden_size,output_size,g=None):
        super(Decoder,self).__init__()
        self.hidden_dim=hidden_size
        self.rnn=nn.GRU(hidden_size,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=1)


    def forward(self,y,hidden):
        output,hidden=self.rnn(y,hidden)
        output=self.softmax(self.out(output[0]))
        return output,hidden


def embed(sentences):
    one_dim_words=[]
    two_dim_words=SPECIAL
    for words in sentences:
        one_dim_words.append(words)
    embed_input=one_dim_words[0::2]
    embed_output=one_dim_words[1::2]
    
    for l in one_dim_words:
        two_dim_words.extend(l)
    two_dim_words=list(set(two_dim_words))
    idx2word=dict(enumerate(two_dim_words))
    word2idx={value:key for key,value in idx2word.items()}
    whole_sentence=' '.join(two_dim_words)

    doc=nlp(whole_sentence)
    for token in doc:
        if token.text in word2idx:
            idx2vec[word2idx[token.text]]=token.vector
        elif token.text in ABBRS:
            idx2vec[word2idx[ABBRS[token.text]]]=token.vector
        else:
            pass
    for idx in idx2word.keys():
        if idx not in idx2vec:
            idx2vec[idx]=idx2vec[word2idx['OOV']]

    embed_input=[[word2idx[word] for word in sentence] for sentence in embed_input]
    embed_output=[[word2idx[word] for word in sentence] for sentence in embed_output]
    
    with open(word2vec_filename,'wb') as f:
        pickle.dump((word2idx,idx2word,idx2vec),f)
    
    #print(word2idx)
    #print(idx2word)
    #print(idx2vec.keys())
    embed_input=torch.LongTensor(embed_input).unsqueeze(2)
    embed_output=torch.LongTensor(embed_output).unsqueeze(2)
    return embed_input,embed_output


def init_model(dataset):
    (input,target)=dataset
    input_size=300
    hidden_size=300
    encoder=Encoder(input_size,hidden_size)

    batch_size=input.size(1)
    decoder=Decoder(hidden_size,len(word2idx))

    loss_fn=nn.NLLLoss()
    max_len=5
    
    q=lambda x:x[-1]
    model=EncoderDecoder(encoder=encoder,
                        decoder=decoder,
                        input=input,
                        output=target,
                        hidden_dim=hidden_size,
                        loss_fn=loss_fn,
                        max_len=max_len,
                        )
    return model


def change():
    if os.path.exists(embed_filename):
        os.remove(embed_filename)
    if os.path.exists(word2vec_filename):
        os.remove(word2vec_filename)

        
load_data()
TEST=True
IGNORE_OUTPUT=False
model=None
if TEST:
    import pickle
    dataset=None
    change()
    if os.path.exists(embed_filename):
        with open(embed_filename,'rb') as f:
            dataset=pickle.load(f)
    else:
        dataset=embed(data)
        with open(embed_filename,'wb') as f:
            pickle.dump(dataset,f)
    with open(word2vec_filename,'rb') as f:
        (word2idx,idx2word,idx2vec)=pickle.load(f)
    model=init_model(dataset)
    model=model.cuda()
    model.check_cuda_params()


import numpy as np
for name,parameters in model.named_parameters():
    print(name,':',parameters.size())
para = sum([np.prod(list(p.size())) for p in model.parameters()])
print(para)
type_size=4
print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))


def train():
    print('start training')
    stop_loss=0.1
    encoder=model.encoder
    decoder=model.decoder
    enc_optimizer=torch.optim.SGD(encoder.parameters(),lr=0.01)
    dec_optimizer=torch.optim.SGD(decoder.parameters(),lr=0.01)
    max_train_step=400
    for step_i in range(max_train_step):
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        loss=model.loss()
        loss.backward(retain_graph=True)
        enc_optimizer.step()
        dec_optimizer.step()
        if not IGNORE_OUTPUT:
            print('step:{0} loss:{1}'.format(step_i,loss))
            if loss<stop_loss:
                print('loss is considerable, stop')
                break
train()