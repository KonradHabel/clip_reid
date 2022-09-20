import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import cv2
import random
import copy
import torch
from torchvision import transforms

class TrainDataset(Dataset):
    def __init__(self,
                 img_path,
                 df,
                 image_transforms=None,
                 prob_flip=0.5,
                 shuffle_batch_size=16):
        
        
        self.img_path = img_path
        self.df = df.set_index("img_id")
        self.image_transforms = image_transforms
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        
        print("\nImages train: {}".format(len(self.df)))
        self.images = self.df.index.values.tolist()
        
        # dict for all images for a given player
        self.player_images = defaultdict(list)
        for img_id in self.images:
            player = self.df.loc[img_id]["player"]
            self.player_images[player].append(img_id)
  
        # dict for all gallery image for a given image
        self.player_images_other = {}
        for img_id in self.images:
            player = self.df.loc[img_id]["player"]
            other_images = copy.deepcopy(self.player_images[player])
            other_images.remove(img_id)
            self.player_images_other[img_id] = np.array(other_images)

        self.samples = copy.deepcopy(self.images)
        self.shuffle()
            
    def __getitem__(self, index):

        # next image (keep in mind to use the custom shuffle instead of Dataloader shuffle)
        img_id_query = self.samples[index]
        img_path_query = "{}/{}/{}.jpeg".format(self.img_path,
                                                self.df.loc[img_id_query]["folder"],
                                                img_id_query)
        
        img_query = cv2.imread(img_path_query)
        img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
        
        if self.image_transforms:
            img_query = self.image_transforms(image=img_query)['image']
        
        
        # random select one other image of the same player as gallery image
        img_id_gallery = np.random.choice(self.player_images_other[img_id_query], 1)[0]
        img_path_gallery = "{}/{}/{}.jpeg".format(self.img_path,
                                                  self.df.loc[img_id_gallery]["folder"],
                                                  img_id_gallery)
    
    
        img_gallery = cv2.imread(img_path_gallery)
        img_gallery = cv2.cvtColor(img_gallery, cv2.COLOR_BGR2RGB)
        
        if self.image_transforms:
            img_gallery = self.image_transforms(image=img_gallery)['image']
  
            
        # player id can be used as label for other loss functions than InfoNCE   
        player = torch.tensor(int(self.df.loc[img_id_query]["player"]), dtype=torch.long)
        
        # random flip both images
        if np.random.random() < self.prob_flip:
            
            img_query = transforms.functional.hflip(img_query)
            img_gallery = transforms.functional.hflip(img_gallery)
        
        return img_query, img_gallery, player
    
    def __len__(self):

        return len(self.samples)


    def shuffle(self):
        '''
        custom shuffle function to prevent having the same player two times in the same batch
        '''
        
        img_ids_select = copy.deepcopy(self.images)
        random.shuffle(img_ids_select)

        batches = []
        players_batch = set()
        break_counter = 0
        
        while True:
            
            if len(img_ids_select) > 0:
                img_id = img_ids_select.pop(0)
                
                player = self.df.loc[img_id]["player"]
                
                if player not in players_batch:
                    players_batch.add(player)
                    batches.append(img_id)
                    
                    # reset break counter
                    break_counter = 0
            
                else:
                    # if img_id already in batch append at end of of the selection
                    img_ids_select.append(img_id)
                    
                    # increase break counter
                    break_counter += 1
                    
                # impossible to add remaining images to batch without having the
                # same player two times in the last batch 
                if break_counter >= 10:
                    break
            
            # no more images left       
            else:
                break
            
            # if batch is filled with unique players reset 
            if len(players_batch) >= self.shuffle_batch_size:
                players_batch = set()
                      
        self.samples = batches
        print("\nShuffle Training Data:")
        print("Lengt Train:", len(batches))
        print("Rest:", len(img_ids_select)) 
        print("First Element: {}".format(self.samples[0]))
        
        
class TestDataset(Dataset):
    def __init__(self,
                 img_path,
                 df,
                 image_transforms=None):
        
        
        self.img_path = img_path
        self.df = df.set_index("img_id")
        self.image_transforms = image_transforms
        self.images = self.df.index.values.tolist()
        
        self.query = []
        self.gallery = []
        self.all = []
        
        for img_id in self.images:
            
            player = self.df.loc[img_id]["player"]
            img_type = self.df.loc[img_id]["img_type"]
            self.all.append((img_id, player, -1))
            
            if img_type == "q":
                self.query.append((img_id, player, 0))
            else:
                self.gallery.append((img_id, player, 1))
        
    def __getitem__(self, index):

        img_id = self.images[index]
        img_path = "{}/{}/{}.jpeg".format(self.img_path,
                                          self.df.loc[img_id]["folder"],
                                          img_id)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.image_transforms:
            img = self.image_transforms(image=img)['image']
             
        player = int(self.df.loc[img_id]["player"])
        
        if self.df.loc[img_id]["img_type"] == "q":
            img_type = 0
        else:
            img_type = 1
            
        
        return img, img_id, player, img_type
    
    def __len__(self):

        return len(self.images)
    
class ChallengeDataset(Dataset):
    def __init__(self,
                 df,
                 image_transforms=None):
        
        
   
        self.df = df.set_index("img_id")
        self.image_transforms = image_transforms
        self.images = self.df.index.values.tolist()
        
        self.query = []
        self.gallery = []
        self.all = []
        
        for img_id in self.images:
            
            player = self.df.loc[img_id]["player"]
            img_type = self.df.loc[img_id]["img_type"]
            self.all.append((img_id, int(player), -1))
            
            if img_type == "q":
                self.query.append((img_id, player, 0))
            else:
                self.gallery.append((img_id, player, 1))
        
    def __getitem__(self, index):

        img_id = self.images[index]
 
        img = cv2.imread(img_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.image_transforms:
            img = self.image_transforms(image=img)['image']
             
        player = int(self.df.loc[img_id]["player"])
        
        if self.df.loc[img_id]["img_type"] == "q":
            img_type = 0
        else:
            img_type = 1
            
        
        return img, img_id, player, img_type
    
    def __len__(self):

        return len(self.images)