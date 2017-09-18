import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

class Encoder(nn.Module):
    def __init__(self,ENC_TYPE,num_rnn_layers=1,rnn_hidden_size=128,dropout=0):
        super(Encoder,self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.ENC_TYPE = ENC_TYPE
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=(3,4),stride=(3,2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=(4,3),stride=(4,2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=(4,2),stride=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.gru = nn.GRU(128,rnn_hidden_size,num_rnn_layers,
                        batch_first=True,
                        dropout=dropout)
        self.linear = nn.Linear(128,rnn_hidden_size*num_rnn_layers)

        self.layer4 = nn.Conv1d(128,256,kernel_size=3,stride=1,padding=1)
        self.layer5 = nn.Conv1d(128,256,kernel_size=3,stride=1,padding=1)
        self.layer6 = nn.Conv1d(128,256,kernel_size=3,stride=1,padding=1)

    def forward(self,x,hidden):
        if self.ENC_TYPE=='CNNRNN':
            h0 = hidden
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out).squeeze()
            out = out.transpose(1,2)
            out,hidden = self.gru(out,h0)
            return out,hidden
        else: #self.ENC_TYPE=='CNN':
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out).squeeze()

            for layer in [self.layer4,self.layer5,self.layer6]:
                input = out
                out = layer(out)
                out = out[:,:128]*nn.functional.sigmoid(out[:,128:])
                out = out + input

            out = out.transpose(1,2)
            hidden = torch.nn.functional.tanh(self.linear(out.mean(dim=1))).view(self.num_rnn_layers,-1,self.rnn_hidden_size)
            return out,hidden

    def initHidden(self,batch_size,use_cuda=False):
        h0 = Variable(torch.zeros(self.num_rnn_layers,batch_size,self.rnn_hidden_size))
        if use_cuda:
            return (h0.cuda())
        else:
            return h0

class RNNDecoder(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, output_size,
                 num_rnn_layers=1, dropout=0.):
        super(RNNDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_vocab_size = input_vocab_size
        self.output_size = output_size

        self.gru = nn.GRU(input_vocab_size, hidden_size,
                          num_rnn_layers, batch_first=True,
                          dropout=dropout)

        #self.grucell = nn.GRUCell(input_size, hidden_size)
        self.out = nn.Linear(hidden_size,output_size)

        self.embedding = nn.Embedding(input_vocab_size, input_vocab_size)
        fix_embedding = torch.from_numpy(np.eye(input_vocab_size, input_vocab_size).astype(np.float32))
        self.embedding.weight = nn.Parameter(fix_embedding)
        self.embedding.weight.requires_grad=False

    def forward(self, input, hidden):
        embed_input = self.embedding(input.unsqueeze(1))
        output,hidden = self.gru(embed_input,hidden)
        output  = self.out(output.squeeze())
        return output, hidden

class Attn(nn.Module):
    def __init__(self,method,hidden_size):
        super(Attn,self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size,hidden_size,bias=False)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size*2,hidden_size,bias=False)
            self.tanh = nn.Tanh()
            self.attn_linear = nn.Linear(hidden_size,1,bias=False)

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: decode hidden state, (batch_size , N)
        :param encoder_outputs: encoder's all states, (batch_size,T,N)
        :return: weithed_context :(batch_size,N), alpha:(batch_size,T)
        """
        hidden_expanded = hidden.unsqueeze(2) #(batch_size,N,1)

        if self.method == 'dot':
            energy = torch.bmm(encoder_outputs,hidden_expanded).squeeze(2)

        elif self.method == 'general':
            energy = self.attn(encoder_outputs)
            energy = torch.bmm(energy,hidden_expanded).squeeze(2)

        elif self.method == 'concat':
            hidden_expanded = hidden.unsqueeze(1).expand_as(encoder_outputs)
            energy = self.attn(torch.cat((hidden_expanded, encoder_outputs), 2))
            energy = self.attn_linear(self.tanh(energy)).squeeze(2)

        alpha = nn.functional.softmax(energy)
        weighted_context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)

        return weighted_context,alpha

class RNNAttnDecoder(nn.Module):
    def __init__(self, attn_model, input_vocab_size, hidden_size,
                 output_size, num_rnn_layers=1, dropout=0.):
        super(RNNAttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_vocab_size = input_vocab_size
        self.output_size = output_size
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
            self.gru = nn.GRU(input_vocab_size + hidden_size, hidden_size,
                              num_rnn_layers, batch_first=True,
                              dropout=dropout)
        else:
            self.attn = None
            self.gru = nn.GRU(input_vocab_size, hidden_size,
                              num_rnn_layers, batch_first=True,
                              dropout=dropout)

        #self.grucell = nn.GRUCell(input_vocab_size + hidden_size, hidden_size)
        self.wc = nn.Linear(2 * hidden_size, hidden_size)#,bias=False)
        self.ws = nn.Linear(hidden_size,output_size)

        self.tanh = nn.Tanh()
        self.embedding = nn.Embedding(input_vocab_size, input_vocab_size)
        fix_embedding = torch.from_numpy(np.eye(input_vocab_size, input_vocab_size).astype(np.float32))
        self.embedding.weight = nn.Parameter(fix_embedding)
        self.embedding.weight.requires_grad=False

    def forward(self, input, last_ht, last_hidden, encoder_outputs):
        '''
        :se
        :param input: (batch_size,)
        :param last_ht: (obatch_size,hidden_size)
        :param last_hidden: (batch_size,hidden_size)
        :param encoder_outputs: (batch_size,T,hidden_size)
        '''
        if self.attn is None:
            embed_input = self.embedding(input.unsqueeze(1))
            output, hidden = self.gru(embed_input, last_hidden)
            output = self.ws(output.squeeze())
            return output, last_ht,hidden,None
        else:
            embed_input = self.embedding(input)
            rnn_input = torch.cat((embed_input,last_ht),1)
            output,hidden = self.gru(rnn_input.unsqueeze(1),last_hidden)
            output = output.squeeze()

            weighted_context, alpha = self.attn(output,encoder_outputs)
            ht = self.tanh(self.wc(torch.cat((output,weighted_context),1)))
            output = self.ws(ht)
            return output,ht,hidden,alpha


class RNNAttnDecoder2(nn.Module):
    def __init__(self,attn_model,input_vocab_size,hidden_size,
                 output_size,num_rnn_layers=1,dropout=0.):
        super(RNNAttnDecoder2,self).__init__()
        self.hidden_size = hidden_size
        self.input_vocab_size = input_vocab_size
        self.output_size = output_size
        self.attn = Attn(attn_model,hidden_size)
        self.gru = nn.GRU(input_vocab_size+hidden_size,hidden_size,
                          num_rnn_layers,batch_first=True,
                          dropout=dropout)
        self.embedding = nn.Embedding(input_vocab_size, input_vocab_size)
        fix_embedding = torch.from_numpy(np.eye(input_vocab_size, input_vocab_size).astype(np.float32))
        self.embedding.weight = nn.Parameter(fix_embedding)
        self.embedding.weight.requires_grad=False

        self.wc = nn.Linear(hidden_size+input_vocab_size, hidden_size)
        self.ws = nn.Linear(hidden_size*2,output_size)

    def forward(self,input,last_ht,last_hidden,encoder_outputs):
        embed_input = self.embedding(input)
        attn_input = self.wc(torch.cat((embed_input,last_hidden[-1]),1))
        weighted_context,alpha = self.attn(attn_input,encoder_outputs)
        rnn_input = torch.cat((embed_input,weighted_context),1)
        output,hidden = self.gru(rnn_input.unsqueeze(1),last_hidden)
        output = output.squeeze()
        output = self.ws(torch.cat((output,nn.functional.tanh(weighted_context)),1))
        return output,last_ht,hidden,alpha

def train(inputs, targets,
          encoder, decoder,
          encoder_optimizer,decoder_optimizer,
          criterion,clip,use_cuda=False,vallina=False):

    loss = 0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    batch_size = inputs.size()[0]
    max_len = targets.size()[1]

    init_hidden = encoder.initHidden(batch_size,use_cuda=use_cuda)
    encoder_outputs,encoder_hidden = encoder(inputs,init_hidden)

    last_hidden = encoder_hidden
    last_ht = Variable(torch.zeros(batch_size, decoder.hidden_size))
    outputs = torch.zeros((batch_size,max_len-1)).long()
    if use_cuda:
        last_ht = last_ht.cuda()
        outputs = outputs.cuda()

    for di in range(max_len-1):
        input = targets[:,di]
        target = targets[:,di+1]
        output,last_ht,last_hidden,alpha = decoder(input,last_ht,last_hidden,encoder_outputs)
        outputs[:,di] = output.max(1)[1].data
        loss += criterion(output,target)

    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    #accuracy = ((targets[:,1:].data==outputs).sum(1)== outputs.size()[1]).sum()/batch_size
    num_eq = (targets[:,1:].data==outputs).sum(dim=1)
    accuracy_clevel = num_eq.sum()/batch_size/(max_len-1)
    accuracy = (num_eq==max_len-1).sum()/batch_size

    return loss.data[0],accuracy_clevel,accuracy

def evaluate(inputs,targets,
             encoder,decoder,criterion,use_cuda=False):
    encoder.train(False)
    decoder.train(False)
    loss = 0

    batch_size = inputs.size()[0]
    max_len = targets.size()[1]

    init_hidden = encoder.initHidden(batch_size,use_cuda=use_cuda)
    encoder_outputs,encoder_hidden = encoder(inputs,init_hidden)

    last_hidden = encoder_hidden
    last_ht = Variable(torch.zeros(batch_size, decoder.hidden_size))
    outputs = torch.zeros((batch_size,max_len-1)).long()
    if use_cuda:
        last_ht = last_ht.cuda()
        outputs = outputs.cuda()


    input = targets[:,0]

    for di in range(max_len-1):
        output, last_ht, last_hidden, alpha = decoder(input, last_ht, last_hidden, encoder_outputs)
        input = output.max(1)[1]
        outputs[:,di] = input.data
        loss += criterion(output,targets[:,di+1])

    #accuracy = ((targets[:,1:].data==outputs).sum(1)== outputs.size()[1]).sum()/batch_size
    num_eq = (targets[:,1:].data==outputs).sum(dim=1)
    accuracy_clevel = num_eq.sum()/batch_size/(max_len-1)
    accuracy = (num_eq==max_len-1).sum()/batch_size

    encoder.train(True)
    decoder.train(True)

    return loss.data[0],accuracy_clevel,accuracy,outputs
