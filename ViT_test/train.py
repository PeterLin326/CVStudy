from vit import ViT
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os 
import torch 
from dataset import MNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

dataset = MNIST()

model = ViT().to(device)

try:
    model.load_state_dict(torch.load('E:/cursor_proj/CV_STUDY/ViT_test/model.pth'))
except:
    pass 

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCH=50
BATCH_SIZE=64

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True,num_workers=10,persistent_workers=True)

iter_count=0
for epoch in range(EPOCH):
    for imgs, labels in dataloader:
        logits = model(imgs.to(device))
        loss = F.cross_entropy(logits, labels.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iter_count%1000==0:
            print("epoch:{},iter:{}, loss:{}".format(epoch, iter_count, loss))
            torch.save(model.state_dict(), 'E:/cursor_proj/CV_STUDY/ViT_test/model.pth')
            #os.replace('.model.pth', 'model.pth')
        iter_count+=1