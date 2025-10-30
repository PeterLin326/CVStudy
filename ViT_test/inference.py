from dataset import MNIST
import matplotlib.pyplot as plt 

import torch 
from vit import ViT
import torch.nn.functional as F 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MNIST()
model = ViT().to(device)
model.load_state_dict(torch.load("/home/lm/workspace/CVStudy/ViT_test/model.pth"))

model.eval()

image,label = dataset[2324]
print("true cls:", label)
plt.imshow(image.permute(1,2,0))
plt.show()

logits = model(image.unsqueeze(0).to(device))
print("predict cls:", logits.argmax(-1).item())