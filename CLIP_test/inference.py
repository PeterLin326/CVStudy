import sys
sys.path.append('/home/lm/workspace/CVStudy/CLIP_test')

from dataset import MNIST
import matplotlib.pyplot as plt 
import torch 
from clip import CLIP
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset=MNIST()

model=CLIP().to(DEVICE)
model.load_state_dict(torch.load('/home/lm/workspace/CVStudy/CLIP_test/mnist-clip/model.pth'))
model.eval()

#1. img cls
image, label = dataset[1]
print("true cls: ",label)
plt.imshow(image.permute(1,2,0))
plt.show()

targets = torch.arange(0,10)
logits=model(image.unsqueeze(0).to(DEVICE), targets.to(DEVICE))
print(logits)
print('CLIP cls:', logits.argmax(-1).item())


#2.image similarity
other_images=[]
other_labels=[]
for i in range(1,101):
    other_image, other_label=dataset[i]
    other_images.append(other_image)
    other_labels.append(other_label)

other_img_embs=model.img_enc(torch.stack(other_images,dim=0).to(DEVICE))

img_emb=model.img_enc(image.unsqueeze(0).to(DEVICE))

logits=img_emb@other_img_embs.T
values,indices=logits[0].topk(10)

plt.figure(figsize=(15,15))
for i,img_idx in enumerate(indices):
    plt.subplot(1, 10, i+1)
    plt.imshow(other_images[img_idx].permute(1,2,0))
    plt.title(other_labels[img_idx])
    plt.axis('off')
    
plt.show()
