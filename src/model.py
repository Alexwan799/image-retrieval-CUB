
from torchvision import models as models
import torch.nn as nn
import torch.nn.functional as F
class EmbeddingModel(nn.Module):

    def __init__(self,embed_dim=128):
        super().__init__()
        backbone=models.resnet50(weights='DEFAULT')
        # cut the original classification head
        backbone.fc=nn.Identity()
        self.backbone=backbone

        self.embedding_head=nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,embed_dim)
            
        )
    
    def forward(self,x):
        x=self.backbone(x)
        x=self.embedding_head(x)
        x=F.normalize(x,p=2,dim=1)
        return x 
    


    
