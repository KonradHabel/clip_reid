import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from clipreid.dataset import ChallengeDataset, TestDataset
from clipreid.evaluator import (
    compute_dist_matrix,
    compute_scores,
    predict,
    write_mat_csv,
)
from clipreid.model import OpenClipModel, TimmModel
from clipreid.transforms import get_transforms
from clipreid.utils import print_line


@dataclass
class Configuration:
    """
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
    """

    # Model
    model: str = (
        "ViT-L-14",
        "openai",
    )  # ('name of Clip model', 'name of dataset') | 'name of Timm model'
    remove_proj = True  # remove projection for Clip ViT models

    # Settings only for Timm models
    img_size: int = (224, 224)  # follow above Link for image size of Timm models
    mean: float = (0.485, 0.456, 0.406)  # mean of ImageNet
    std: float = (0.229, 0.224, 0.225)  # std  of ImageNet

    # Eval
    batch_size: int = 64  # batch size for evaluation
    normalize_features: int = True  # L2 normalize of features during eval

    # Split for Eval
    fold: int = -1  # -1 for given test split | int >=0 for custom folds

    # Checkpoints: str or tuple of str for ensemble (checkpoint1, checkpoint2, ...)
    checkpoints: str = "./model/ViT-L-14_openai/fold-1_seed_1/weights_e4.pth"
    # "./model/ViT-L-14_openai/all_data_seed_1/weights_e4.pth")

    # checkpoints: str = "./model/ViT-L-14_openai/Paper/weights_e4.pth"

    # Dataset
    data_dir: str = "./data/data_wyscout_reid"

    # show progress bar
    verbose: bool = True

    # set num_workers to 0 if OS is Windows
    num_workers: int = 0 if os.name == "nt" else 8

    # use GPU if available
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------------------------------------------------#
# Config                                                                                                               #
# ----------------------------------------------------------------------------------------------------------------------#
config = Configuration()

# ----------------------------------------------------------------------------------------------------------------------#
# Model                                                                                                                #
# ----------------------------------------------------------------------------------------------------------------------#
print("\nModel: {}".format(config.model))

if isinstance(config.model, tuple):

    model = OpenClipModel(
        config.model[0], config.model[1], remove_proj=config.remove_proj
    )

    img_size = model.get_image_size()

    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)


else:
    model = TimmModel(config.model, pretrained=True)

    img_size = config.img_size
    mean = config.mean
    std = config.std


dist_matrix_list = []
dist_matrix_rerank_list = []

if not isinstance(config.checkpoints, list) and not isinstance(
    config.checkpoints, tuple
):
    checkpoints = [config.checkpoints]
else:
    checkpoints = config.checkpoints


for checkpoint in checkpoints:

    print_line(name=checkpoint, length=80)

    # load pretrained Checkpoint
    model_state_dict = torch.load(checkpoint)
    model.load_state_dict(model_state_dict, strict=True)

    # Model to device
    model = model.to(config.device)

    print("\nImage Size:", img_size)
    print("Mean: {}".format(mean))
    print("Std:  {}".format(std))

    # ------------------------------------------------------------------------------------------------------------------#
    # DataLoader                                                                                                       #
    # ------------------------------------------------------------------------------------------------------------------#

    # Transforms
    val_transforms, train_transforms = get_transforms(img_size, mean, std)

    # Dataframes
    df = pd.read_csv("./data/data_wyscout_reid/train_df.csv")
    df_challenge = pd.read_csv("./data/data_wyscout_reid/challenge_df.csv")

    if config.fold == -1:
        # Use given test split
        df_train = df[df["split"] == "train"]
        df_test = df[df["split"] == "test"]
    else:
        # Use custom folds
        df_train = df[df["fold"] != config.fold]
        df_test = df[df["fold"] == config.fold]

    # Validation
    test_dataset = TestDataset(
        img_path="./data/data_wyscout_reid", df=df_test, image_transforms=val_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # Challenge
    challenge_dataset = ChallengeDataset(
        df=df_challenge, image_transforms=val_transforms
    )

    challenge_loader = DataLoader(
        challenge_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # ------------------------------------------------------------------------------------------------------------------#
    # Test                                                                                                             #
    # ------------------------------------------------------------------------------------------------------------------#
    print_line(name="Eval Fold: {}".format(config.fold), length=80)

    # extract features
    features_dict = predict(
        model,
        dataloader=test_loader,
        device=config.device,
        normalize_features=config.normalize_features,
        verbose=config.verbose,
    )

    # compute distance matrix
    dist_matrix_test, dist_matrix_test_rerank = compute_dist_matrix(
        features_dict, test_dataset.query, test_dataset.gallery, rerank=True
    )

    # without re-ranking
    print("\nwithout re-ranking:")
    mAP = compute_scores(
        dist_matrix_test, test_dataset.query, test_dataset.gallery, cmc_scores=True
    )

    save_path = "{}/test_dmat.csv".format("/".join(checkpoint.split("/")[:-1]))
    print("write distance matrix:", save_path)
    write_mat_csv(save_path, dist_matrix_test, test_dataset.query, test_dataset.gallery)

    # save with re-ranking
    print("\nwith re-ranking:")
    mAP = compute_scores(
        dist_matrix_test_rerank,
        test_dataset.query,
        test_dataset.gallery,
        cmc_scores=True,
    )

    save_path = "{}/test_dmat_rerank.csv".format("/".join(checkpoint.split("/")[:-1]))
    print("write distance matrix:", save_path)
    write_mat_csv(
        save_path, dist_matrix_test_rerank, test_dataset.query, test_dataset.gallery
    )

    # ------------------------------------------------------------------------------------------------------------------#
    # Challenge                                                                                                        #
    # ------------------------------------------------------------------------------------------------------------------#
    print_line(name="Challenge", length=80)

    # extract features
    features_dict = predict(
        model,
        dataloader=challenge_loader,
        device=config.device,
        normalize_features=config.normalize_features,
        verbose=config.verbose,
    )

    # compute distance matrix
    dist_matrix, dist_matrix_rerank = compute_dist_matrix(
        features_dict, challenge_dataset.query, challenge_dataset.gallery, rerank=True
    )

    # collect for ensemble
    dist_matrix_list.append(dist_matrix)
    dist_matrix_rerank_list.append(dist_matrix_rerank)

    # save without re-ranking
    save_path = "{}/challenge_dmat.csv".format("/".join(checkpoint.split("/")[:-1]))
    print("write distance matrix:", save_path)
    write_mat_csv(
        save_path, dist_matrix, challenge_dataset.query, challenge_dataset.gallery
    )

    # save with re-ranking
    save_path = "{}/challenge_dmat_rerank.csv".format(
        "/".join(checkpoint.split("/")[:-1])
    )
    print("write distance matrix:", save_path)
    write_mat_csv(
        save_path,
        dist_matrix_rerank,
        challenge_dataset.query,
        challenge_dataset.gallery,
    )


# ----------------------------------------------------------------------------------------------------------------------#
# Ensemble                                                                                                             #
# ----------------------------------------------------------------------------------------------------------------------#
if len(dist_matrix_list) > 1:

    print_line(name="Ensemble", length=80)

    # without re-ranking
    dist_matrix_ensemble = np.stack(dist_matrix_list, axis=0).mean(0)
    save_path = "{}/challenge_dmat_ensemble.csv".format(
        "/".join(checkpoint.split("/")[:-2])
    )
    print("write distance matrix:", save_path)
    write_mat_csv(
        save_path,
        dist_matrix_ensemble,
        challenge_dataset.query,
        challenge_dataset.gallery,
    )

    # with re-ranking
    dist_matrix_rerank_ensemble = np.stack(dist_matrix_rerank_list, axis=0).mean(0)
    save_path = "{}/challenge_dmat_rerank_ensemble.csv".format(
        "/".join(checkpoint.split("/")[:-2])
    )
    print("write distance matrix:", save_path)
    write_mat_csv(
        save_path,
        dist_matrix_rerank_ensemble,
        challenge_dataset.query,
        challenge_dataset.gallery,
    )
