import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.optim import Optimizer
import os

dict={'BOS':0}

'batch_first===False'
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,embed_input,embed_output,hidden_dim,loss,max_len,q,batch_first=True):
        'embed_input:batch_size*sourceL*dim'
        'assume embed_input,embed_output are torch variables'
        super(EncoderDecoder,self).__init__() 
        self.encoder=encoder
        self.decoder=decoder
        batch_size=embed_input.size(1)
        self.enc_len=embed_input.size(0)
        self.dec_len=embed_output.size(0)
        self.hidden_dim=hidden_dim
        self.encode_hid=nn.Parameter(torch.FloatTensor(1,batch_size,hidden_dim))
        self.decode_hid=nn.Parameter(torch.FloatTensor(1,batch_size,hidden_dim))
        'parameter is changing, the data should be saved in the list'
        nn.init.xavier_uniform_(self.encode_hid,gain=0.01)
        nn.init.xavier_uniform_(self.decode_hid,gain=0.01)
        self.predict_y_list=[]
        self.encode_hid_list=[self.encode_hid.data]
        self.decode_hid_list=[self.decode_hid.data]
        self.input=embed_input
        self.output=embed_output
        self.loss_fn=loss
        self.q=q


    def forward(self):
        #print(type(self.encode_hid))
        #for i in self.encode_hid_list:
        #    print(i[0,0,:10])
        for enc_i in range(self.enc_len):
            _,self.encode_hid.data=self.encoder(self.input,self.encode_hid)
            self.encode_hid_list.append(self.encode_hid.data)
        c=self.q(self.encode_hid_list)
        y=self.encode_hid_list[-1]
        for dec_i in range(self.dec_len):
            y,self.decode_hid.data=self.decoder(y,c,self.decode_hid)
            self.predict_y_list.append(y)
            self.decode_hid_list.append(self.decode_hid.data)
        return output


    def loss(self):
        input=self.input
        output=self.forward()
        loss_var=self.loss_fn(input,output)
        return loss_var


    def predict(self,x):
        output=self.encoder(x,)


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
    def __init__(self,batch_size,input_size,hidden_size,g=None):
        super(Decoder,self).__init__()
        self.dim=input_size
        self.hidden_dim=hidden_size
        self.rnn=nn.GRU(input_size,hidden_size)
        self.Wc=nn.Linear(hidden_size,hidden_size)
        self.Wh=nn.Linear(hidden_size,hidden_size)


    def forward(self,y,c,hidden):
        output,hidden=self.rnn(y,hidden)
        'c.size==hidden.size=1*batch_size*hidden_size'
        hidden=self.Wc(c)+self.Wh(hidden)
        return output,hidden


def embed(sentences):
    'return sourceL*batch_size*dim'
    import spacy
    nlp=spacy.load('en_core_web_md')
    input_vectors=[]
    for sentence in sentences:
        doc=nlp(sentence)
        input_vectors.append([token.vector for token in doc])
    input_vectors=torch.FloatTensor(input_vectors).permute(1,0,2)
    'feature_dim is 300'
    return input_vectors


def train(dataset):
    (input,target)=dataset
    input_size=input.size(-1)
    #print(input.size())
    hidden_size=input_size
    encoder=Encoder(input_size,hidden_size)

    batch_size=input.size(1)
    decoder=Decoder(batch_size,input_size,hidden_size)

    loss=torch.nn.MSELoss(reduction='mean')
    max_len=3
    stop_loss=0.1
    q=lambda x:x[-1]
    model=EncoderDecoder(encoder=encoder,
                        decoder=decoder,
                        embed_input=input,
                        embed_output=target,
                        hidden_dim=hidden_size,
                        loss=loss,
                        max_len=max_len,
                        q=q)
    
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
    for step_i in range(max_len):
        optimizer.zero_grad()
        loss=model.loss()
        loss.backward()
        optimizer.step()
        print('step:{0} loss:{1}'.format(step_i,loss))
        if loss<stop_loss:
            print('loss is considerable, stop')
            break


def change():
    os.remove(test.file)


inputs=['a test sentence','repeat pad pad']
outputs=['yeah a test sentence','repeat pad pad pad']
if __name__=='__main__':
    import pickle
    embed_filename='test.file'
    test=True
    if test:
        #change()
        if os.path.exists(embed_filename):
            with open(embed_filename,'rb') as f:
                dataset=pickle.load(f)
        else:
            dataset=embed(inputs),embed(outputs)
            with open(embed_filename,'wb') as f:
                pickle.dump(dataset,f)
        train(dataset)
