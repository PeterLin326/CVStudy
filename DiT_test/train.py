from config import *
from torch.utils.data import DataLoader
from dataset import MNIST
from diffusion import forward_add_noise
import torch 
from torch import nn 
import os 
from dit import Dit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MNIST()
model = Dit(img_size=28,patch_size=4,channel=1,emb_size=64,label_num=10,dit_num=3,head=4).to(device)

# try:
#     model.load_state_dict(torch.load('/home/lm/workspace/CVStudy/Dit_test/model.pth'))
# except:
#     pass

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn=nn.L1Loss()

EPOCH=500
BATCH_SIZE=300

dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)
model.train()

iter_count=0
for epoch in range(EPOCH):
    for imgs,labels in dataloader:
        x=imgs*2-1
        t=torch.randint(0,T,(imgs.size(0),))
        y=labels
        
        x,noise=forward_add_noise(x,t)
        pred_noise=model(x.to(device),t.to(device),y.to(device))
        
        loss=loss_fn(pred_noise,noise.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iter_count % 1000==0:
            print("epoch:{}, iter:{}, loss:{}".format(epoch,iter_count,loss))
            torch.save(model.state_dict(),"/home/lm/workspace/CVStudy/Dit_test/model.pth")
        iter_count+=1
        
