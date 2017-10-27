from data_utils_torch import *
from ctcmodel import *
from warpctc_pytorch import CTCLoss
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    BATCH_SIZE = 64
    HEIGHT = 48
    WIDTH = 128
    HIDDEN_SIZE = 128
    NUM_RNN_LAYERS = 1
    DROPOUT = 0.2
    LR = 0.0003
    CLIP = 10.
    NUM_EPOCHS = 100
    PRINT_EVERY_N_ITER = 100
    SAVE_DIR = 'CTC128_lr0.0003cp10'
    if not os.path.exists("results"):
        os.mkdir("results")
    SAVE_DIR = os.path.join("results", SAVE_DIR)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    TOTAL_SIZE = None
    TEST_SIZE = 8192

    USE_CUDA = torch.cuda.is_available()

    dl_train, dl_test, vocab = load_dataset(batch_size=BATCH_SIZE, test_size=TEST_SIZE,total_size=TOTAL_SIZE)

    END = vocab['token2id']['$']
    VOCAB_SIZE = len(vocab['token2id'])-2 #exclude "$" and " ", and use "^" as <BLANK>

    ctc = CTCModel(output_size=VOCAB_SIZE,num_rnn_layers=NUM_RNN_LAYERS,
                   rnn_hidden_size=HIDDEN_SIZE,
                   dropout=DROPOUT)

    if USE_CUDA:
        ctc.cuda()

    ctc_params = list(filter(lambda p: p.requires_grad, ctc.parameters()))
    ctc_optimizer = optim.Adam(ctc_params, lr=LR)
    criterion = CTCLoss()

    epoch_train_loss = []
    epoch_train_accuracy = []
    batch_train_loss = []
    batch_train_accuracy = []
    test_loss = []
    test_accuracy = []
    for epoch in range(1, NUM_EPOCHS + 1):

        loss = accuracy = 0
        batches_loss = batches_acc = 0
        for num_iter, (x, y) in enumerate(dl_train):
            vx = Variable(x)
            y = y[:,1:].type(torch.IntTensor)
            vy = Variable(y[y<END-1].contiguous())
            lens = Variable(torch.from_numpy(
                (np.where(y.numpy()==END)[1]).astype(np.int32)))

            a_loss, a_accuracy = CTCtrain(vx,vy,lens,ctc, ctc_optimizer,
                                    criterion, CLIP, use_cuda=USE_CUDA)
            loss += a_loss
            accuracy += a_accuracy
            batches_loss += a_loss
            batches_acc += a_accuracy

            if (num_iter + 1) % PRINT_EVERY_N_ITER == 0:
                batches_loss /= PRINT_EVERY_N_ITER
                batches_acc /= PRINT_EVERY_N_ITER
                print("Iteration: {}/{} Epoch: {}/{}".format(
                    num_iter + 1, len(dl_train), epoch, NUM_EPOCHS))
                print("recent batches:\n"
                      "loss {}\n"
                      "accuracy {}".format(batches_loss, batches_acc))
                batch_train_loss.append(batches_loss)
                batch_train_accuracy.append(batches_acc)
                batches_loss = batches_acc = 0

        epoch_train_loss.append(loss / len(dl_train))
        epoch_train_accuracy.append(accuracy / len(dl_train))
        print("epoch train loss: {}\n epoch train accuracy: {}".format(epoch_train_loss[-1], epoch_train_accuracy[-1]))

        # test
        loss = accuracy = 0
        for num_iter, (x, y) in enumerate(dl_test):
            vx = Variable(x)
            y = y[:,1:].type(torch.IntTensor)
            vy = Variable(y[y<END-1].contiguous())
            lens = Variable(torch.from_numpy(
                (np.where(y.numpy()==END)[1]).astype(np.int32)))

            a_loss, a_accuracy, outputs = CTCevaluate(vx, vy, lens,ctc, criterion,CLIP,
                                                    use_cuda=USE_CUDA)
            loss += a_loss
            accuracy += a_accuracy
        test_loss.append(loss / len(dl_test))
        test_accuracy.append(accuracy / len(dl_test))
        print("test loss: {}\n"
              "test accuracy: {}".format(test_loss[-1], test_accuracy[-1]))

        c = np.random.choice(BATCH_SIZE)
        print(''.join(vocab['id2token'][i] for i in outputs[c]) + '|' + ''.join(
            vocab['id2token'][i] for i in y[c]) + '|')


    print("Training over")
    # save figures
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(batch_train_loss, 'r', label='loss')
    ax1.legend()
    ax2.plot(batch_train_accuracy, label='acc')
    ax2.legend()
    fig.savefig(os.path.join(SAVE_DIR, "sampled_batch_error.png"))
    print("A figure is saved.")

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot(epoch_train_loss, 'r', label='train_loss')
    ax1.plot(test_loss, 'b', label='test_loss')
    ax1.legend()
    ax2.plot(epoch_train_accuracy, 'r', label='train_acc')
    ax2.plot(test_accuracy, 'b', label='test_acc')
    ax2.legend()
    fig.savefig(os.path.join(SAVE_DIR, "epoch_error.png"))
    print("Another fig is saved.")
    # plt.show()
    np.savetxt(os.path.join(SAVE_DIR, "acc.txt"),
               np.vstack([epoch_train_accuracy, test_accuracy]).T)
