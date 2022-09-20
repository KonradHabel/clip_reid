import time
from tqdm import tqdm
from .utils import AverageMeter
from torch.cuda.amp import autocast
import torch
from transformers import get_polynomial_decay_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup

def train(model,
          dataloader,
          loss_function,
          optimizer,
          device,
          scheduler=None,
          scaler=None,
          gradient_accumulation=1,
          gradient_clipping=None,
          verbose=True,
          multi_gpu=False):

    # set model train mode
    model.train()
    
    losses = AverageMeter()
    
    # wait a second bevor starting progress bar
    time.sleep(1)
    
    # Zero gradients for first step
    optimizer.zero_grad()
    
    step = 1
    
    if verbose:
        bar = tqdm(dataloader,
                   total=len(dataloader),
                   ascii=True,
                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                   desc="Train")
    else:
        bar = dataloader
        
    
    # for loop over one epoch
    for query, gallery, ids in bar:
         
        # data (batches) to device   
        query = query.to(device)
        gallery =  gallery.to(device)

        if scaler:
            with autocast():
                
                # Forward pass
                features1, features2 = model(query, gallery)
                
                if multi_gpu:
                    loss = loss_function(features1, features2, model.module.model.logit_scale.exp())
                else:
                    loss = loss_function(features1, features2, model.model.logit_scale.exp())
                
                losses.update(loss.item())
                
                  
            scaler.scale(loss).backward()
            
            if gradient_clipping is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping) 

            if step % gradient_accumulation == 0:

                # Update model parameters (weights)
                scaler.step(optimizer)
                scaler.update()

                # Zero gradients for next step
                optimizer.zero_grad()
                
                # Scheduler
                if scheduler is not None:
                    scheduler.step()
   
        else:

            # Forward pass
            features1, features2 = model(query, gallery)
            
            if multi_gpu:
                loss = loss_function(features1, features2, model.module.model.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.model.logit_scale.exp())
          
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            if gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)  
            
            if step % gradient_accumulation == 0:

                # Update model parameters (weights)
                optimizer.step()

                # Zero gradients for next step
                optimizer.zero_grad()
                
                # Scheduler
                if scheduler is not None:
                    scheduler.step()
        
        
        if verbose:
            monitor = {"loss": "{:.2f}".format(losses.val),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if verbose:
        bar.close()
  
    return losses.avg


def get_scheduler(train_config, optimizer, train_loader_length):
    
    train_steps = int((train_loader_length * train_config.epochs) / train_config.gradient_accumulation)
    warmup_steps = int(train_loader_length * train_config.warmup_epochs)
    print("\nWarmup Epochs: {} - Warmup Steps: {}".format(train_config.warmup_epochs, warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(train_config.epochs, train_steps)) 
       
    if train_config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(train_config.lr, train_config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                              num_training_steps=train_steps,
                                                              lr_end = train_config.lr_end,
                                                              power=2,
                                                              num_warmup_steps=warmup_steps)
    elif train_config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(train_config.lr))   

        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif train_config.scheduler == "linear":
        print("\nScheduler: linear - max LR: {}".format(train_config.lr))
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_training_steps=train_steps,
                                                    num_warmup_steps=warmup_steps)
        
    elif train_config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(train_config.lr))
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=warmup_steps)
        
    else:
        print("\nScheduler: None")
        scheduler = None
        
    return scheduler
           
