from clip import CLIP
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import torch 
from dataset import MNIST

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MNIST()
model = CLIP().to(DEVICE)

try:
    model.load_state_dict(torch.load('model.pth'))
except:
    pass 

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

ITER_BATCH_COUNT=100000
BATCH_SIZE=64
TARGET_COUNT=10

dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)

for i in range(ITER_BATCH_COUNT):
    while True:
        imgs,labels = next(iter(dataloader))
        if torch.unique(labels).shape[0]<TARGET_COUNT:
            continue
        target = set()
        indices=[]
        for j in range(BATCH_SIZE):
            if labels[j].item() in target:
                continue
            target.add(labels[j].item())
            indices.append(j)
            if len(target)==TARGET_COUNT:
                break
        imgs = imgs[indices]
        labels=labels[indices]
        break
    
    logits=model(imgs.to(DEVICE), labels.to(DEVICE))

    targets = torch.arange(0, TARGET_COUNT).to(DEVICE)
    loss_i=F.cross_entropy(logits,targets)
    loss_t=F.cross_entropy(logits.permute(1,0), targets)
    loss=(loss_i+loss_t)/2

    optimizer.zero_grad()

    if i %1000==0:
        print('iter:{},loss:{}'.format(i,loss))
        torch.save(model.state_dict(),'/home/lm/workspace/CVStudy/CLIP_test/model.pth')