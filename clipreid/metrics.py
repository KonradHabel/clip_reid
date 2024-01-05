"""
Source: https://github.com/DeepSportRadar/player-reidentification-challenge

"""

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool_)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def cmc(
    distmat,
    query_ids=None,
    gallery_ids=None,
    query_cams=None,
    gallery_cams=None,
    topk=100,
    separate_camera_set=False,
    single_gallery_shot=False,
    first_match_break=False,
):

    distmat = to_numpy(distmat)
    m, n = distmat.shape

    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)

    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = gallery_ids[indices] == query_ids[:, np.newaxis]

    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = (gallery_ids[indices[i]] != query_ids[i]) | (
            gallery_cams[indices[i]] != query_cams[i]
        )
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= gallery_cams[indices[i]] != query_cams[i]
        if not np.any(matches[i, valid]):
            continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = valid & _unique_sample(ids_dict, len(valid))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1.0 / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk:
                    break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(
    distmat, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None
):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = gallery_ids[indices] == query_ids[:, np.newaxis]
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = (gallery_ids[indices[i]] != query_ids[i]) | (
            gallery_cams[indices[i]] != query_cams[i]
        )
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true):
            continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def map_at_r(
    distmat, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None
):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = gallery_ids[indices] == query_ids[:, np.newaxis]

    # Compute mAP@R for each query
    map_r_scores = []
    for i in range(m):
        # Get the total number of relevant items (R)
        R = np.sum(gallery_ids == query_ids[i])

        # Filter out the same id and same camera
        valid = (gallery_ids[indices[i]] != query_ids[i]) | (
            gallery_cams[indices[i]] != query_cams[i]
        )

        # Compute precision at i
        precision_at_i = []
        for j in range(R):
            if valid[j] and matches[i, indices[i][j]]:
                precision = np.sum(matches[i, indices[i][: j + 1]]) / (j + 1)
                precision_at_i.append(precision)
            else:
                precision_at_i.append(0)

        # Compute score for this query
        map_r_score = np.sum(precision_at_i) / R
        map_r_scores.append(map_r_score)

    return np.mean(map_r_scores)
