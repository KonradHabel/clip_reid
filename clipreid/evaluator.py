"""
Source: https://github.com/DeepSportRadar/player-reidentification-challenge

for: pairwise_distance, compute_scores, write_mat_csv

"""

import time
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .metrics import cmc, map_at_r, mean_ap
from .rerank import re_ranking


def predict(model, dataloader, device, normalize_features=False, verbose=True):

    # wait a second bevor starting progress bar
    time.sleep(1)

    model.eval()

    if verbose:
        bar = tqdm(
            dataloader,
            total=len(dataloader),
            ascii=True,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            desc="Test ",
        )
    else:
        bar = dataloader

    features = OrderedDict()

    file_names = []
    players = []
    img_types = []

    for img, file_name, player, img_type in bar:

        img = img.to(device)

        file_names.extend(file_name)
        players.extend(player)
        img_types.extend(img_type)

        with torch.no_grad():
            output = model(img)
            if normalize_features:
                output = F.normalize(output, p=2, dim=1)
            output = output.cpu()

        for i in range(len(output)):
            features[file_name[i]] = output[i]

    if verbose:
        bar.close()

    return features


def pairwise_distance(features, query=None, gallery=None):

    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    dist = (
        torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    )
    dist.addmm_(x, y.t(), beta=1, alpha=-2)

    return dist


def compute_dist_matrix(
    features_dict, query, gallery, rerank=False, k1=20, k2=6, lambda_value=0.3
):

    dist_matrix = pairwise_distance(features_dict, query, gallery)

    if rerank:
        distmat_qq = pairwise_distance(features_dict, query, query)
        distmat_gg = pairwise_distance(features_dict, gallery, gallery)
        dist_matrix_rerank = re_ranking(
            dist_matrix, distmat_qq, distmat_gg, k1=k1, k2=k2, lambda_value=lambda_value
        )
        return dist_matrix.numpy(), dist_matrix_rerank
    else:
        return dist_matrix


def compute_scores(
    distmat,
    query=None,
    gallery=None,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    cmc_topk=(1, 5, 10),
    cmc_scores=True,
):

    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (
            query_ids is not None
            and gallery_ids is not None
            and query_cams is not None
            and gallery_cams is not None
        )

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    mAP_at_R = map_at_r(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print("mAP: {:4.2%}".format(mAP))
    print("mAP@R: {:4.2%}".format(mAP_at_R))

    if cmc_scores:
        # Compute all kinds of CMC scores
        cmc_configs = {
            "allshots": dict(
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=False,
            ),
            "cuhk03": dict(
                separate_camera_set=True,
                single_gallery_shot=True,
                first_match_break=False,
            ),
            "market1501": dict(
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True,
            ),
        }
        cmc_scores = {
            name: cmc(
                distmat, query_ids, gallery_ids, query_cams, gallery_cams, **params
            )
            for name, params in cmc_configs.items()
        }

        print("CMC Scores{:>12}{:>12}{:>12}".format("allshots", "cuhk03", "market1501"))
        for k in cmc_topk:
            print(
                "  top-{:<4}{:12.2%}{:12.2%}{:12.2%}".format(
                    k,
                    cmc_scores["allshots"][k - 1],
                    cmc_scores["cuhk03"][k - 1],
                    cmc_scores["market1501"][k - 1],
                )
            )

    return mAP


def write_mat_csv(fpat, dist_matrix, query, gallery):
    gallery_order_list = [pid for _, pid, _ in gallery]
    query_order_list = [pid for _, pid, _ in query]
    data = np.array([0, *gallery_order_list])
    rows = np.array(query_order_list)[:, np.newaxis]
    with open(fpat, "w") as f:
        np.savetxt(f, data.reshape(1, data.shape[0]), delimiter=",", fmt="%i")
        np.savetxt(
            f,
            np.hstack((rows, dist_matrix)),
            newline="\n",
            fmt=["%i", *["%10.5f"] * dist_matrix.shape[1]],
            delimiter=",",
        )
