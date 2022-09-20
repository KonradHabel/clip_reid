import torch
import timm
import open_clip
import numpy as np
import torch.nn as nn
              
class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True):
        super(TimmModel, self).__init__()
        
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0) 
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            
    def forward(self, img1, img2=None):
        
        if img2 is not None:
            images = torch.cat([img1, img2], dim=0)
            image_features = self.model(images)
            
            image_features1 = image_features[:len(img1), :]
            image_features2 = image_features[len(img1):, :]
            
            return image_features1, image_features2
            
        else:
            image_features = self.model(img1)
            
        return image_features
    
class OpenClipModel(nn.Module):

    def __init__(self,
                 model_name,
                 pretrained,
                 remove_proj=False):
        super(OpenClipModel, self).__init__()
        
        self.model_name = model_name
        
        self.model = open_clip.create_model(model_name,
                                            pretrained=pretrained)  
         
        # delete text parts of clip model
        del(self.model.transformer)
        del(self.model.token_embedding)
        del(self.model.ln_final)
        del(self.model.positional_embedding)
        del(self.model.text_projection)
        
        if remove_proj and "ViT" in model_name:
            width, output_dim = self.model.visual.proj.shape
            print("Remove Projection Layer - old output size: {} - new output size: {}".format(output_dim, width))
            self.model.visual.proj = None

    def set_grad_checkpoint(self, enable=True): 
        if "ViT" in self.model_name:
            self.model.visual.set_grad_checkpointing(enable)
            print("Use Gradient Checkpointing for {}".format(self.model_name))
        else:
            print("Gradient Checkpointing not available for {}".format(self.model_name))
        
    def get_image_size(self):
            return self.model.visual.image_size
    
    def forward(self, img1, img2=None):
        
        if img2 is not None:
            images = torch.cat([img1, img2], dim=0)
            image_features = self.model.encode_image(images)
            
            image_features1 = image_features[:len(img1), :]
            image_features2 = image_features[len(img1):, :]
            
            return image_features1, image_features2
            
        else:
            image_features = self.model.encode_image(img1)
            
        return image_features