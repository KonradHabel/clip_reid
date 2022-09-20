import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipLoss(nn.Module):

    def __init__(self,
                 loss_function,
                 device):
        
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, query_features, gallery_features, logit_scale):
        
        query_features = F.normalize(query_features, dim=-1)
        gallery_features = F.normalize(gallery_features, dim=-1)
        
        logits_per_query = logit_scale * query_features @ gallery_features.T
        
        logits_per_gallery = logits_per_query.T
        
        labels = torch.arange(len(logits_per_query), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_query , labels) + self.loss_function(logits_per_gallery, labels))/2

        return loss  