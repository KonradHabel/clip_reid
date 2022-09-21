import os
import sys
import shutil
import torch
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import DataLoader

from clipreid.loss import ClipLoss
from clipreid.trainer import train, get_scheduler
from clipreid.utils import Logger, setup_system, print_line
from clipreid.model import TimmModel, OpenClipModel
from clipreid.transforms import get_transforms
from clipreid.dataset import TrainDataset, TestDataset
from clipreid.evaluator import predict, compute_dist_matrix, compute_scores

@dataclass
class Configuration:
    '''
    --------------------------------------------------------------------------
    Open Clip Models:
    --------------------------------------------------------------------------    
    - ('RN50', 'openai')
    - ('RN50', 'yfcc15m')
    - ('RN50', 'cc12m')
    - ('RN50-quickgelu', 'openai')
    - ('RN50-quickgelu', 'yfcc15m')
    - ('RN50-quickgelu', 'cc12m')
    - ('RN101', 'openai')
    - ('RN101', 'yfcc15m')
    - ('RN101-quickgelu', 'openai')
    - ('RN101-quickgelu', 'yfcc15m')
    - ('RN50x4', 'openai')
    - ('RN50x16', 'openai')
    - ('RN50x64', 'openai')
    - ('ViT-B-32', 'openai')
    - ('ViT-B-32', 'laion2b_e16')
    - ('ViT-B-32', 'laion400m_e31')
    - ('ViT-B-32', 'laion400m_e32')
    - ('ViT-B-32-quickgelu', 'openai')
    - ('ViT-B-32-quickgelu', 'laion400m_e31')
    - ('ViT-B-32-quickgelu', 'laion400m_e32')
    - ('ViT-B-16', 'openai')
    - ('ViT-B-16', 'laion400m_e31')
    - ('ViT-B-16', 'laion400m_e32')
    - ('ViT-B-16-plus-240', 'laion400m_e31')
    - ('ViT-B-16-plus-240', 'laion400m_e32')
    - ('ViT-L-14', 'openai')
    - ('ViT-L-14', 'laion400m_e31')
    - ('ViT-L-14', 'laion400m_e32')
    - ('ViT-L-14-336', 'openai')
    - ('ViT-H-14', 'laion2b_s32b_b79k')
    - ('ViT-g-14', 'laion2b_s12b_b42k')
    --------------------------------------------------------------------------
    Timm Models:
    --------------------------------------------------------------------------
    - 'convnext_base_in22ft1k'
    - 'convnext_large_in22ft1k'
    - 'vit_base_patch16_224'
    - 'vit_large_patch16_224'
    - ...
    - https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv
    --------------------------------------------------------------------------
    '''

    # Model
    model: str = ('ViT-L-14', 'openai')   # ('name of Clip model', 'name of dataset') | 'name of Timm model'
    remove_proj = True                    # remove projection for Clip ViT models
    
    # Settings only for Timm models 
    img_size: int = (224, 224)            # follow above Link for image size of Timm models
    mean:   float = (0.485, 0.456, 0.406) # mean of ImageNet
    std:    float = (0.229, 0.224, 0.225) # std  of ImageNet
    
    # Split
    train_on_all: bool = False            # True: train incl. test data  
    fold: int = -1                        # -1 for given test split | int >=0 for custom folds 
    
    # Training 
    seed: int = 1                         # seed for Python, Numpy, Pytorch
    epochs: int = 4                       # epochs to train
    batch_size: int = 16                  # batch size for training
    batch_size_eval: int = 64             # batch size for evaluation
    gpu_ids: tuple = (0,)                 # GPU ids for training e.g. (0,1) multi GPU 
    mixed_precision: bool = True          # fp16 for faster training
    
    # Learning Rate
    lr: float = 0.00004                   # use 4 * 10^-5 for ViT | 4 * 10^-4 for CNN
    scheduler: str = "polynomial"         # "polynomial" | "cosine" | "linear" | "constant" | None
    warmup_epochs: float = 1.0            # linear increase lr
    lr_end: float = 0.00001               # only for "polynomial"
    
    # Optimizer  
    gradient_clipping: float = None       # None | float
    grad_checkpointing: bool = False      # gradient checkpointing for CLIP ViT models
    gradient_accumulation: int = 1        # 1: no gradient accumulation
    
    # Loss
    label_smoothing: float = 0.1          # label smoothing for crossentropy loss
    
    # Eval
    zero_shot: bool = True                # eval before training
    rerank: bool = True                   # use re-ranking as post-processing
    normalize_features: int = True        # L2 normalize of features during eval
    
    # Dataset
    data_dir: str = "./data/data_reid"    # datset path
    prob_flip: str = 0.5                  # probability for random horizontal flip during training 
         
    # Savepath for model checkpoints
    model_path: str = "./model"          
    
    # Checkpoint to start from
    checkpoint_start: str = None         
    
    # show progress bar
    verbose: bool = True 
  
    # set num_workers to 0 on Windows
    num_workers: int = 0 if os.name == 'nt' else 8  
    
    # train on GPU if available
    device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    
    # for better performance
    cudnn_benchmark: bool = True
    
    # make cudnn deterministic
    cudnn_deterministic: bool = True      # set to False for faster training of CNNs
     
#----------------------------------------------------------------------------------------------------------------------#  
# Config                                                                                                               #
#----------------------------------------------------------------------------------------------------------------------#  
config = Configuration()

if isinstance(config.model, tuple):
    # Clip models
    if config.train_on_all:
        model_path = "{}/{}_{}/all_data_seed_{}".format(config.model_path,
                                               config.model[0],
                                               config.model[1],
                                               config.seed)
    else:
        model_path = "{}/{}_{}/fold{}_seed_{}".format(config.model_path,
                                               config.model[0],
                                               config.model[1],
                                               config.fold,
                                               config.seed)
else:
    # Timm models
    if config.train_on_all:
        model_path = "{}/{}/all_data_seed_{}".format(config.model_path,
                                                     config.model,
                                                     config.seed)
    else:
        model_path = "{}/{}/fold{}_seed_{}".format(config.model_path,
                                            config.model,
                                            config.fold,
                                            config.seed)

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))

# Redirect print to both console and log file
sys.stdout = Logger("{}/log.txt".format(model_path))

# Set seed
setup_system(seed=config.seed,
             cudnn_benchmark=config.cudnn_benchmark,
             cudnn_deterministic=config.cudnn_deterministic)

#----------------------------------------------------------------------------------------------------------------------#  
# Model                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------#  
print("\nModel: {}".format(config.model))

if isinstance(config.model, tuple):

    model = OpenClipModel(config.model[0],
                          config.model[1],
                          remove_proj=config.remove_proj
                          )
    
    img_size = model.get_image_size()
    
    mean=(0.48145466, 0.4578275, 0.40821073)
    std=(0.26862954, 0.26130258, 0.27577711)
    
    if config.grad_checkpointing: 
        model.set_grad_checkpoint(enable=config.grad_checkpointing)
       
else:
    model = TimmModel(config.model,
                      pretrained=True)

    img_size = config.img_size
    mean = config.mean
    std = config.std
    
    
    
# load pretrained Checkpoint    
if config.checkpoint_start is not None:  
    print("\nStart from:", config.checkpoint_start)
    model_state_dict = torch.load(config.checkpoint_start)  
    model.load_state_dict(model_state_dict, strict=True)
    
# Data parallel
print("\nGPUs available:", torch.cuda.device_count())  
if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
    print("Using Data Prallel with GPU IDs: {}".format(config.gpu_ids))
    model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)    
    multi_gpu = True
else:
    multi_gpu = False  
    
# Model to device   
model = model.to(config.device)

print("\nImage Size:", img_size)
print("Mean: {}".format(mean))
print("Std:  {}".format(std)) 

#----------------------------------------------------------------------------------------------------------------------#  
# DataLoader                                                                                                           #
#----------------------------------------------------------------------------------------------------------------------#  
# Data
df = pd.read_csv("{}/train_df.csv".format(config.data_dir))

# Split data
if config.train_on_all:
    df_train = df
    df_test = df[df["split"] == "test"]
else:
    if config.fold == -1:
        # Use given test split
        df_train = df[df["split"] == "train"]
        df_test = df[df["split"] == "test"]
    else:
        # Use custom folds
        df_train = df[df["fold"] != config.fold]
        df_test = df[df["fold"] == config.fold]

  
# Transforms
val_transforms, train_transforms = get_transforms(img_size, mean, std)

# Train
train_dataset = TrainDataset(img_path=config.data_dir,
                             df=df_train,
                             image_transforms=train_transforms,
                             prob_flip=config.prob_flip,
                             shuffle_batch_size=config.batch_size)

train_loader = DataLoader(train_dataset,
                          batch_size=config.batch_size,
                          num_workers=config.num_workers,
                          shuffle=False,
                          pin_memory=True,
                          drop_last=True)

# Validation
test_dataset = TestDataset(img_path=config.data_dir,
                           df=df_test,
                           image_transforms=val_transforms)


test_loader = DataLoader(test_dataset,
                         batch_size=config.batch_size_eval,
                         num_workers=config.num_workers,
                         shuffle=False,
                         pin_memory=True)

#----------------------------------------------------------------------------------------------------------------------#  
# Loss                                                                                                                 #
#----------------------------------------------------------------------------------------------------------------------#  
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
loss_function = ClipLoss(loss_function=loss_fn,
                         device=config.device)

#----------------------------------------------------------------------------------------------------------------------#  
# optimizer and scaler                                                                                                 #
#----------------------------------------------------------------------------------------------------------------------#  
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

if config.mixed_precision:
    scaler = torch.cuda.amp.GradScaler(init_scale=2.**10)
else:
    scaler = None
    
#----------------------------------------------------------------------------------------------------------------------#  
# Scheduler                                                                                                            #
#----------------------------------------------------------------------------------------------------------------------#  
if config.scheduler is not None:
    scheduler = get_scheduler(config,
                              optimizer,
                              train_loader_length=len(train_loader))       
else:
    scheduler = None
   
    
#----------------------------------------------------------------------------------------------------------------------#  
# Zero Shot                                                                                                            #
#----------------------------------------------------------------------------------------------------------------------#  
if config.zero_shot:
    
    print_line(name="Zero-Shot", length=80)
    
    features_dict = predict(model,
                            dataloader=test_loader,
                            device=config.device,
                            normalize_features=config.normalize_features,
                            verbose=config.verbose)
    
    dist_matrix, dist_matrix_rerank = compute_dist_matrix(features_dict, 
                                                          test_dataset.query,
                                                          test_dataset.gallery,
                                                          rerank=True)
    
    print("\nWithout re-ranking:")
    mAP = compute_scores(dist_matrix,
                         test_dataset.query,
                         test_dataset.gallery)
    
    if dist_matrix_rerank is not None:
        print("\nWith re-ranking:")
        mAP = compute_scores(dist_matrix_rerank,
                             test_dataset.query,
                             test_dataset.gallery)
        

#----------------------------------------------------------------------------------------------------------------------#  
# Train                                                                                                                #
#----------------------------------------------------------------------------------------------------------------------#  
for epoch in range(1, config.epochs+1):

    print_line(name="Epoch: {}".format(epoch), length=80)
    
    # Train
    train_loss = train(model,
                       dataloader=train_loader,
                       loss_function=loss_function,
                       optimizer=optimizer,
                       device=config.device,
                       scheduler=scheduler,
                       scaler=scaler,
                       gradient_accumulation=config.gradient_accumulation,
                       gradient_clipping=config.gradient_clipping,
                       verbose=config.verbose,
                       multi_gpu=multi_gpu)

    
    print("Avg. Train Loss = {:.4f} - Lr = {:.6f}\n".format(train_loss,
                                                           optimizer.param_groups[0]['lr']))
    # Evaluate
    features_dict = predict(model,
                            dataloader=test_loader,
                            device=config.device,
                            normalize_features=config.normalize_features,
                            verbose=config.verbose)
    
    dist_matrix, dist_matrix_rerank = compute_dist_matrix(features_dict, 
                                                          test_dataset.query,
                                                          test_dataset.gallery,
                                                          rerank=True)
    
    print("\nWithout re-ranking:")
    mAP = compute_scores(dist_matrix,
                         test_dataset.query,
                         test_dataset.gallery)
    
    if dist_matrix_rerank is not None:
        print("\nWith re-ranking:")
        mAP_rerank = compute_scores(dist_matrix_rerank,
                                    test_dataset.query,
                                    test_dataset.gallery)
        
    checkpoint_path = '{}/weights_e{}.pth'.format(model_path, epoch)
            
    # Save model  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), checkpoint_path)
    else:
        torch.save(model.state_dict(), checkpoint_path)
    
    # Shuffle data for next epoch
    train_loader.dataset.shuffle()        

    
    





    
    
    


    
    