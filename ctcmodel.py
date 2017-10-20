import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
from utils import  decode_ctc_outputs

class CTCModel(nn.Module):
    def __init__(self,output_size,rnn_hidden_size=128, num_rnn_layers=1, dropout=0):
        super(CTCModel, self).__init__()
        self.num_rnn_layers = num_rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.output_size = output_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 4), stride=(3, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(4, 3), stride=(4, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(4, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.gru = nn.GRU(128, rnn_hidden_size, num_rnn_layers,
                          batch_first=True,
                          dropout=dropout)
        self.linear = nn.Linear(rnn_hidden_size,output_size)

    def forward(self, x, hidden):
        h0 = hidden
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out).squeeze()
        out = out.transpose(1, 2)
        out, hidden = self.gru(out, h0)
        out = self.linear(out)
        return out

    def initHidden(self,batch_size,use_cuda=False):
        h0 = Variable(torch.zeros(self.num_rnn_layers,batch_size,self.rnn_hidden_size))
        if use_cuda:
            return (h0.cuda())
        else:
            return h0

def CTCtrain(inputs,targets,lens,ctc,ctc_optimizer,criterion,clip,use_cuda=False):
    loss = 0
    ctc_optimizer.zero_grad()
    batch_size = inputs.size()[0]
    init_hidden = ctc.initHidden(batch_size,use_cuda=use_cuda)
    ctc_outputs = ctc(inputs,init_hidden)

    ctcloss_inputs = ctc_outputs.transpose(0,1) #SeqLen * BatchSize * Hidden
    label_lens = lens
    act_lens = Variable(torch.IntTensor(batch_size*[ctc_outputs.size()[1]]),requires_grad=False)

    loss = criterion(ctcloss_inputs,targets,act_lens,label_lens)

    loss.backward()
    torch.nn.utils.clip_grad_norm(ctc.parameters(), clip)
    ctc_optimizer.step()

    #TODO
    decoded_outputs = decode_ctc_outputs(ctc_outputs)
    decoded_targets = np.split(targets.data.numpy(),lens.data.numpy().cumsum())[:-1]
    accuracy = np.array([np.array_equal(decoded_targets[i],decoded_outputs[i])
                         for i in range(batch_size)]).mean()


    return loss.data[0],accuracy

def CTCevaluate(inputs,targets,lens,ctc,criterion,clip,use_cuda=False):
    ctc.train(False)
    loss = 0
    batch_size = inputs.size()[0]
    init_hidden = ctc.initHidden(batch_size,use_cuda=use_cuda)
    ctc_outputs = ctc(inputs,init_hidden)

    ctcloss_inputs = ctc_outputs.transpose(0,1) #SeqLen * BatchSize * Hidden
    label_lens = lens
    act_lens = Variable(torch.IntTensor(batch_size*[ctc_outputs.size()[1]]),requires_grad=False)

    loss = criterion(ctcloss_inputs,targets,act_lens,label_lens)

    #TODO
    decoded_outputs = decode_ctc_outputs(ctc_outputs)
    decoded_targets = np.split(targets.data.numpy(),lens.data.numpy().cumsum())[:-1]
    accuracy = np.array([np.array_equal(decoded_targets[i],decoded_outputs[i])
                         for i in range(batch_size)]).mean()

    ctc.train(True)
    return loss.data[0],accuracy,decoded_outputs