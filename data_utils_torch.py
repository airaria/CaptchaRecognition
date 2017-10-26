import numpy as np
from torch.utils.data import TensorDataset,DataLoader
import torch
import os,pickle
def load_dataset(batch_size,dir='data',n_workers=0,test_size=16384,total_size=None):
    print ("Loading data...")
    data = np.load(os.path.join(dir,'captcha.npz'))
    image = data['img'].astype(np.float32)/127.5-1
    text = data['text']
    print ("Loading dictionary...")
    vocab = pickle.load(open(os.path.join(dir,'captcha.vocab_dict'),'rb'),encoding='utf8')

    print ("Convert to tensor...")
    if total_size is None:
        image = torch.Tensor(image).permute(0,3,1,2)
        text = torch.LongTensor(text)
    else:
        image = torch.Tensor(image[:total_size]).permute(0,3,1,2)
        text = torch.LongTensor(text[:total_size])

    image_train = image[:-test_size]
    image_test = image[-test_size:]
    text_train = text[:-test_size]
    text_test = text[-test_size:]
    print ("Build dataset...")
    dataset_train = TensorDataset(image_train,text_train)
    dataset_test = TensorDataset(image_test,text_test)

    if torch.cuda.is_available():
        pm = True
    else:
        pm = False
    print ("Build dataloader...")
    dataloader_train = DataLoader(dataset_train,batch_size,True,num_workers=n_workers)
    dataloader_test = DataLoader(dataset_test,batch_size,shuffle=False)
    print ("data ready!")
    return dataloader_train,dataloader_test,vocab

if __name__=='__main__':
    dl_train,dl_test,vocab = load_dataset(32)
