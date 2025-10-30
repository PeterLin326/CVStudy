from torch import nn
import torch

#print(torch.__version__)

class ViT(nn.Module):
    def __init__(self, emb_size=16):
        super().__init__()
        self.patch_size = 4
        self.patch_count = 28 // self.patch_size
       # according to nlp->bert(encoder)!!!
        self.conv = nn.Conv2d(in_channels=1, out_channels=self.patch_size**2,kernel_size=self.patch_size, padding=0, stride=self.patch_size)
        self.patch_emb = nn.Linear(in_features = self.patch_size**2, out_features=emb_size)
        self.cls_token = nn.Parameter(torch.rand(1,1,emb_size))
        self.pos_emb = nn.Parameter(torch.rand(1, self.patch_count**2+1, emb_size))
        #num_layers = norm + mutiheadatt + shortcut + norm + MLP...   
        self.transformer_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=emb_size, nhead=2, batch_first = True), num_layers=3)
        self.cls_linear = nn.Linear(in_features=emb_size, out_features=10)
        
    def forward(self, x):
        x = self.conv(x) #(B,C,W,H)
        #print("after conv layer:",x.size())
        x = x.view(x.size(0), x.size(1), self.patch_count**2) #(B,C,W*H)
        #print(x.size())
        x = x.permute(0,2,1) #(B,W*H,C)
        #print(x.size())
        x=self.patch_emb(x) #(B, W*H,emb_size)
        #print("after patch emb:", x.size())
        cls_token = self.cls_token.expand(x.size(0), 1, x.size(2)) #(B,1,emb_size)
        
        x = torch.cat((cls_token,x), dim=1)
        x = self.pos_emb+x
        
        y = self.transformer_enc(x)
        return self.cls_linear(y[:,0,:])
    
if __name__ == '__main__':
    vit=ViT()
    x = torch.rand(5,1,28,28)
    y = vit(x)
    print(y.shape)
    print(y)
        

        
        