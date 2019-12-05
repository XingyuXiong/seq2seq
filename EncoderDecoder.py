import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.optim import Optimizer
import os


#word2vec={}
word_list=[]
word2idx={}
idx2word={}
idx2vec={}
embed_filename='test.file'
word2vec_filename='dict.file'

'batch_first===False'
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,input,output,hidden_dim,loss,max_len,q,batch_first=True):
        'input:batch_size*sourceL*dim'
        'assume embed_input,embed_output are torch variables'
        super(EncoderDecoder,self).__init__() 
        self.encoder=encoder
        self.decoder=decoder
        self.batch_size=input.size(1)
        self.enc_len=input.size(0)
        self.dec_len=output.size(0)
        self.hidden_dim=hidden_dim
        'parameter is changing, the data should be saved in the list'
        self.predict_y_list=[]
        self.encode_hid_list=[torch.rand(1,self.batch_size,hidden_dim)*0.01]
        self.decode_hid_list=[torch.rand(1,self.batch_size,hidden_dim)*0.01]
        self.decode_output_list=[]
        self.input=input
        self.output=output
        self.loss_fn=loss
        self.crossentropy=nn.CrossEntropyLoss(reduction=False)
        self.q=q
        self.word_score_dis=torch.Tensor(self.dec_len,self.batch_size,len(word_list
            ))
        'convert hidden_size-length output to dictionary-length scores of words'
        self.W=torch.rand(self.batch_size,len(word_list),self.hidden_dim)
        #'used when softmax in different batches'
        #self.curr_batch=None


    def forward(self):
        #print(type(self.encode_hid))
        #for i in self.encode_hid_list:
        #    print(i[0,0,:10])
        encode_hid=self.encode_hid_list[0]
        dec_init_input=None
        for enc_i in range(self.enc_len):
            print(self.input.size())
            print(encode_hid.size())
            dec_init_input,encode_hid=self.encoder(self.input,encode_hid)
            self.encode_hid_list.append(encode_hid.data)
        c=self.q(self.encode_hid_list)
        self.decode_output_list.append(dec_init_input)
        output=dec_init_input
        decode_hid=self.decode_hid_list[0]
        for dec_i in range(self.dec_len):
            'output.size()==sourceL*batch_size*hidden_dim'
            output,decode_hid=self.decoder(output,c,decode_hid)
            self.decode_output_list.append(output)
            self.cal_score(dec_i,output)
            #self.decode_hid_list.append(self.decode_hid)


    def loss(self):
        self.forward()   
        loss_var=0
        for output in self.decode_output_list:
            pass
        return loss_var


    def predict(self,x):
        #output=self.encoder(x,)
        pass


    def cal_score(self,dec_i,output):
        #print(self.W.size())
        step_output=output[min(dec_i,self.enc_len-1)].unsqueeze(-1)
        #print(step_output.size())
        #for i in range(self.batch_size):
        self.word_score_dis[dec_i]=torch.bmm(self.W,step_output).squeeze()


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
        'self-define layer'
        hidden=self.Wc(c)+self.Wh(hidden)
        return output,hidden


def embed(sentences):
    'return sourceL*batch_size*dim'
    '''
    import spacy
    nlp=spacy.load('en_core_web_md')
    input_idxs=[]
    for sentence in sentences:
        doc=nlp(sentence)      
        i=0
        for token in doc:
            #word2vec[token.text]=token.vector
            word2idx[token.text]=i
            idx2word[i]=token.text
            idx2vec[token.text]=i
            i+=1
        input_idxs.append([token.vector for token in doc])
    input_idxs=torch.FloatTensor(input_idxs).permute(1,0,2)
    #print(dic)
    with open(word2vec_filename,'wb') as f:
        pickle.dump(dic,f)
    'feature_dim is 300'
    return input_idxs
    '''   
    'two local lists'
    word_list2=[]
    word_list3=[]
    for sentence in sentences:
        words=sentence.split()
        word_list2.append(words)
    input_idxs=word_list2[0::2]
    output_idxs=word_list2[1::2]

    for l in word_list2:
        word_list3.extend(l)
    word_list3=list(set(word_list3))
    idx2word=dict(enumerate(word_list3))
    word2idx={value:key for key,value in idx2word.items()}
    whole_sentence=' '.join(word_list3)

    print(word2idx)
    input_idxs=[[word2idx[word] for word in sentence] for sentence in input_idxs]
    output_idxs=[[word2idx[word] for word in sentence] for sentence in output_idxs]
    print(input_idxs[0])
    import spacy
    nlp=spacy.load('en_core_web_md')
    doc=nlp(whole_sentence)
    for token in doc:
        idx2vec[word2idx[token.text]]=token.vector
    with open(word2vec_filename,'wb') as f:
        pickle.dump((word_list3,word2idx,idx2word,idx2vec),f)    
    
    return (torch.Tensor(input_idxs),torch.Tensor(output_idxs))


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
                        input=input,
                        output=target,
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
    if os.path.exists(embed_filename):
        os.remove(embed_filename)
    if os.path.exists(word2vec_filename):
        os.remove(word2vec_filename)


data=['a test sentence','yeah a test sentence','repeat pad pad','repeat pad pad pad']
if __name__=='__main__':
    import pickle
    test=True
    dataset=None
    if test:
        change()
        if os.path.exists(embed_filename):
            with open(embed_filename,'rb') as f:
                dataset=pickle.load(f)
            with open(word2vec_filename,'rb') as f:
                (word_list,word2idx,idx2word,idx2vec)=pickle.load(f)
        else:
            dataset=embed(data)
            with open(embed_filename,'wb') as f:
                pickle.dump(dataset,f)
        train(dataset)
