import torch
import torch.nn.functional as F
import torch.nn as nn
class MyTripletMarginLoss(nn.Module):

    def __init__(self,margin=0.3):
        super().__init__()
        self.margin=margin 
    

    def forward(self,embeddings:torch.Tensor,labels:torch.Tensor)-> torch.Tensor:

        sq_norm=torch.sum(embeddings**2,dim=1)

        dist_sq=sq_norm.unsqueeze(1)+sq_norm.unsqueeze(0)-(embeddings@embeddings.T)*2

        distable=torch.sqrt(torch.clamp(dist_sq,min=1e-12))

        positive_mask=labels.unsqueeze(1)==labels.unsqueeze(0)
        negative_mask=~positive_mask

        positive_mask.fill_diagonal_(False)

        anchor_positive_dist=distable * positive_mask
        hardest_positive_dist,_=anchor_positive_dist.max(dim=1)

        anchor_negative_dist = distable + (~negative_mask).float() * 1e9
        hardest_negative_dist,_=anchor_negative_dist.min(dim=1)

        loss=F.relu(hardest_positive_dist-hardest_negative_dist+self.margin).mean()

        return loss
        
        
        
class MyProxyNCA(nn.Module):
    
    def __init__(self,classes_num, embed_dim ):
        super().__init__()
        self.classes_num=classes_num
        self.embed_dim=embed_dim
        self.proxies=nn.Parameter(torch.rand(classes_num,embed_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        self.register_buffer("proxy_labels", torch.arange(classes_num))

        
    def forward(self,embeddings,labels):
        
        proxies = F.normalize(self.proxies, p=2, dim=1) 
        dist_table=self.get_dist(embeddings,proxies)
        proxy_labels= self.proxy_labels
        positive_mask=(labels.unsqueeze(1)==proxy_labels.unsqueeze(0))
        negative_mask=~positive_mask
        p_val=torch.sum(dist_table * positive_mask, dim=1)
        n_val=torch.sum(torch.exp(dist_table*-1)*negative_mask,dim=1)
        n_val=torch.clamp(n_val,min=1e-12)
        loss=(p_val+torch.log(n_val)).mean()
        
        return loss
        
        
        
        
        
    
    def get_dist(self,embed_a,embed_b):
        
        norm_a=torch.sum(embed_a **2, dim=1)
        norm_b=torch.sum(embed_b **2, dim=1)
        dist=norm_a.unsqueeze(1)+norm_b.unsqueeze(0)-2 * embed_a@embed_b.T
        dist=torch.sqrt(torch.clamp(dist,1e-12))
        return dist
    

      
        
    
    