import pandas as pd
from glob import glob
from sklearn.model_selection import GroupKFold
from dataclasses import dataclass

@dataclass
class Configuration:
    n_folds: int = 10
    # path: str = "./data/data_reid"
    path: str = "./data/data_wyscout_reid"

#----------------------------------------------------------------------------------------------------------------------#  
# Config                                                                                                               #
#----------------------------------------------------------------------------------------------------------------------# 
config = Configuration()

#----------------------------------------------------------------------------------------------------------------------#  
# Get image for train, test and challenge                                                                              #
#----------------------------------------------------------------------------------------------------------------------# 

# train
train = glob("{}/reid_training/*.jpeg".format(config.path))
print("Train:{}".format(len(train)))

# test
test_query = glob("{}/reid_test/query/*.jpeg".format(config.path))
test_gallery = glob("{}/reid_test/gallery/*.jpeg".format(config.path))
print("Test Query: {} - Test Gallery: {}".format(len(test_query), len(test_gallery)))

# challenge
challenge_query = glob("{}/reid_challenge/query/*.jpeg".format(config.path))
challenge_gallery = glob("{}/reid_challenge/gallery/*.jpeg".format(config.path))
print("Challenge Query: {} - Challenge Gallery: {}".format(len(challenge_query), len(challenge_gallery)))


#----------------------------------------------------------------------------------------------------------------------#  
# Train: Query + Gallery                                                                                               #
#----------------------------------------------------------------------------------------------------------------------#
img_id = []
folder = []
player = []
game = []
split = []
img_type = []

for f in train:
    
    data = f.replace("\\", "/").split("/")[-1].split(".")[0]
    img_id.append(data)
    data = data.split("_")
            
    p = data[0]
    g = data[1]
    i = data[2]
    
    folder.append("reid_training")
    player.append(p)
    game.append("train_{}".format(g))
    split.append("train")
    
    if i == "00":
        img_type.append("q")
    else:
        img_type.append("g")
    
#----------------------------------------------------------------------------------------------------------------------#  
# Test: Query                                                                                                          #
#----------------------------------------------------------------------------------------------------------------------#
for f in test_query:
    
    data = f.replace("\\", "/").split("/")[-1].split(".")[0]
    img_id.append(data)
    data = data.split("_")
            
    p = data[0]
    g = data[1]
    
    folder.append("reid_test/query")
    player.append(p)
    game.append("test_{}".format(g))
    split.append("test")
    img_type.append("q")
    
#----------------------------------------------------------------------------------------------------------------------#  
# Test: Gallery                                                                                                        #
#----------------------------------------------------------------------------------------------------------------------#
for f in test_gallery:
    
    data = f.replace("\\", "/").split("/")[-1].split(".")[0]
    img_id.append(data)
    data = data.split("_")
            
    p = data[0]
    g = data[1]
    
    folder.append("reid_test/gallery")
    player.append(p)
    game.append("test_{}".format(g))
    split.append("test")
    img_type.append("g")   

#----------------------------------------------------------------------------------------------------------------------#  
# Dataframe for Train + Test                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#
df_train = pd.DataFrame({"img_id": img_id,
                         "folder": folder,
                         "player": player,
                         "game": game,
                         "split": split,
                         "img_type": img_type,
                         })

df_train["fold"] = -1

# CV splits beside offical split
cv = GroupKFold(n_splits=config.n_folds)
split = list(cv.split(df_train, df_train['player'], df_train['game']))

for i in range(config.n_folds):
    train_idx, val_idx = split[i]
    df_train.loc[val_idx, "fold"] = i
    
    
# save train DataFrame 
df_train.to_csv("{}/train_df.csv".format(config.path), index=False)
    
#----------------------------------------------------------------------------------------------------------------------#  
# Challenge: Query                                                                                                     #
#----------------------------------------------------------------------------------------------------------------------#
img_id = []
img_type = []
player = []

for f in challenge_query:
    
    data = f.replace("\\", "/")
    
    img_id.append(data)
    player.append(data.split("/")[-1].split(".")[0])
    img_type.append("q")
    
#----------------------------------------------------------------------------------------------------------------------#  
# Challenge: Gallery                                                                                                   #
#----------------------------------------------------------------------------------------------------------------------#
for f in challenge_gallery:
    
    data = f.replace("\\", "/")
    
    img_id.append(data)
    player.append(data.split("/")[-1].split(".")[0])    
    img_type.append("g")  
    
#----------------------------------------------------------------------------------------------------------------------#  
# Dataframe for Challenge                                                                                              #
#----------------------------------------------------------------------------------------------------------------------#
df_challenge = pd.DataFrame({"img_id": img_id,
                             "player": player,
                             "img_type": img_type,
                             })    

# save challenge DataFrame    
df_challenge.to_csv("{}/challenge_df.csv".format(config.path), index=False)    
