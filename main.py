from data_utils_torch import *
from model import *
import os
import matplotlib.pyplot as plt
if __name__ == '__main__':
    BATCH_SIZE = 64
    TEST_SIZE = 8192
    HEIGHT = 48
    WIDTH = 128
    HIDDEN_SIZE = 128
    NUM_RNN_LAYERS = 2
    DROPOUT = 0
    LR = 0.0003
    CLIP=10.
    NUM_EPOCHS = 50
    PRINT_EVERY_N_ITER = 100
    ATTN_TYPE='dot'
    ATTN_CLASS='type1' #type1 (Luong) | type2
    ENC_TYPE='CNNRNN' #CNN|CNNRNN
    SAVE_DIR ='CNNRNNdot128_lr0.0003cp10type1'
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    USE_CUDA = torch.cuda.is_available()

    dl_train, dl_test, vocab = load_dataset(batch_size=BATCH_SIZE,test_size=TEST_SIZE)

    VOCAB_SIZE = len(vocab['token2id'])

    encoder = Encoder(ENC_TYPE,num_rnn_layers=NUM_RNN_LAYERS,
                            rnn_hidden_size=HIDDEN_SIZE,
                            dropout=DROPOUT)

    if ATTN_CLASS=='type1':
        decoder = RNNAttnDecoder(ATTN_TYPE,input_vocab_size=VOCAB_SIZE,hidden_size=HIDDEN_SIZE,
                                 output_size=VOCAB_SIZE,num_rnn_layers=NUM_RNN_LAYERS,
                                 dropout=DROPOUT)
    elif ATTN_CLASS=='type2':
        decoder = RNNAttnDecoder2(ATTN_TYPE,input_vocab_size=VOCAB_SIZE,hidden_size=HIDDEN_SIZE,
                                  output_size=VOCAB_SIZE,num_rnn_layers=NUM_RNN_LAYERS,
                                  dropout=DROPOUT)
    else:
        raise NotImplementedError

    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    '''
    decoder_vallina = RNNDecoder(input_size=VOCAB_SIZE,hidden_size=HIDDEN_SIZE,
                                 output_size=VOCAB_SIZE,num_rnn_layers=2,
                                 dropout=0.)
    '''

    encoder_params = list(filter(lambda p:p.requires_grad,encoder.parameters()))
    decoder_params = list(filter(lambda p:p.requires_grad,decoder.parameters()))
    encoder_optimizer = optim.Adam(encoder_params, lr=LR)
    decoder_optimizer = optim.Adam(decoder_params, lr=LR)
    criterion = nn.CrossEntropyLoss()

    epoch_train_loss = []
    epoch_train_accclevel = []
    epoch_train_accuracy = []
    batch_train_loss = []
    batch_train_accclevel=[]
    batch_train_accuracy = []
    test_loss =[]
    test_accclevel=[]
    test_accuracy = []
    for epoch in range(1,NUM_EPOCHS+1):

        loss = accuracy = accclevel = 0
        batches_loss = batches_acc = batches_acccl= 0
        for num_iter,(x,y) in enumerate(dl_train):
            vx = Variable(x)
            vy = Variable(y)
            if USE_CUDA:
                vx = vx.cuda()
                vy = vy.cuda()
            a_loss,a_accclevel,a_accuracy = train(vx,vy,
                                   encoder,decoder,
                                   encoder_optimizer,decoder_optimizer,
                                   criterion,CLIP,use_cuda=USE_CUDA)
            loss += a_loss
            accuracy += a_accuracy
            accclevel += a_accclevel
            batches_loss += a_loss
            batches_acc += a_accuracy
            batches_acccl += a_accclevel

            if (num_iter+1)%PRINT_EVERY_N_ITER == 0:
                batches_loss/=PRINT_EVERY_N_ITER
                batches_acc/=PRINT_EVERY_N_ITER
                batches_acccl/=PRINT_EVERY_N_ITER
                print ("Iteration: {}/{} Epoch: {}/{}".format(
                    num_iter+1,len(dl_train),epoch, NUM_EPOCHS))
                print ("recent batches:\n"
                       "loss {}\n"
                       "accuracy {} accclevel {}".format(batches_loss,batches_acc,batches_acccl))
                batch_train_loss.append(batches_loss)
                batch_train_accuracy.append(batches_acc)
                batch_train_accclevel.append(batches_acccl)
                batches_loss=batches_acc=batches_acccl=0

        epoch_train_loss.append(loss/len(dl_train))
        epoch_train_accuracy.append(accuracy/len(dl_train))
        epoch_train_accclevel.append(accclevel/len(dl_train))
        print("epoch train loss: {}\n"
              "epoch train accuracy: {} accclevel {}".format(epoch_train_loss[-1],epoch_train_accuracy[-1],epoch_train_accclevel[-1]))

        #test
        loss = accuracy = accclevel = 0
        for num_iter,(x,y) in enumerate(dl_test):
            vx = Variable(x)
            vy = Variable(y)
            if USE_CUDA:
                vx = vx.cuda()
                vy = vy.cuda()
            a_loss,a_accclevel,a_accuracy,outputs = evaluate(vx,vy,encoder,decoder,criterion,use_cuda=USE_CUDA)
            loss += a_loss
            accuracy += a_accuracy
            accclevel += a_accclevel
        test_loss.append(loss/len(dl_test))
        test_accclevel.append(accclevel/len(dl_test))
        test_accuracy.append(accuracy/len(dl_test))
        print("test loss: {}\n"
              "test accuracy: {} accclevel {}".format(test_loss[-1],test_accuracy[-1],test_accclevel[-1]))

        c = np.random.choice(BATCH_SIZE)
        print(''.join(vocab['id2token'][i] for i in outputs[c])+'|'+''.join(vocab['id2token'][i] for i in y[c][1:])+'|')


    print ("Training over")
    # save figures
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(batch_train_loss,'r',label='loss')
    ax1.legend()
    ax2.plot(batch_train_accuracy,label='acc')
    ax2.plot(batch_train_accclevel,label='acccl')
    ax2.legend()
    fig.savefig(os.path.join(SAVE_DIR,"sampled_batch_error.png"))
    print ("A figure is saved.")
    
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(epoch_train_loss,'r',label='train_loss')
    ax1.plot(test_loss,'b',label='test_loss')
    ax1.legend()
    ax2.plot(epoch_train_accuracy,'r',label='train_acc')
    ax2.plot(test_accuracy,'b',label='test_acc')
    ax2.plot(test_accclevel,'g',label='test_acccl')
    ax2.legend()
    fig.savefig(os.path.join(SAVE_DIR,"epoch_error.png"))
    print ("Another fig is saved.")
    #plt.show()
    np.savetxt(os.path.join(SAVE_DIR,"acc.txt"),np.vstack([epoch_train_accuracy,test_accuracy,epoch_train_accclevel,test_accclevel]).T)
    '''
    #test data flow
    x,y = next(train_iter)
    vx = Variable(x)
    vy = Variable(y)
    initHidden = encoder.initHidden(BATCH_SIZE)


    loss = train(vx, vy, encoder, decoder,encoder_optimizer, decoder_optimizer,
              criterion, clip=CLIP)

    encoder_outputs,encoder_hidden = encoder(vx,initHidden)

    last_ht = Variable(torch.zeros(BATCH_SIZE, decoder.hidden_size))
    decoder_output, ht, hidden, alpha= decoder(input=vy[:,0].long(),
                                            last_ht = last_ht,
                                            last_hidden = encoder_hidden,
                                            encoder_outputs = encoder_outputs)
    '''
