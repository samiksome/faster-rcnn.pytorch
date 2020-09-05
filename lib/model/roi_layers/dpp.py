import sys

import torch
import numpy as np

from model import _CppDPP


def dpp(proposals_single, scores_single, dpp_alpha, dpp_k):
    # compute IoU matrix
    proposals_pairs = torch.cat((proposals_single.repeat_interleave(
        proposals_single.shape[0], dim=0), proposals_single.repeat(proposals_single.shape[0], 1)), dim=1)

    intersections_x1 = proposals_pairs[:, [0, 4]].max(dim=1)[0]
    intersections_y1 = proposals_pairs[:, [1, 5]].max(dim=1)[0]
    intersections_x2 = proposals_pairs[:, [2, 6]].min(dim=1)[0]
    intersections_y2 = proposals_pairs[:, [3, 7]].min(dim=1)[0]

    intersections_areas = (intersections_x2-intersections_x1+1).clamp(min=0) * \
        (intersections_y2-intersections_y1+1).clamp(min=0)
    proposals_areas_1 = (proposals_pairs[:, 2]-proposals_pairs[:, 0]+1)*(proposals_pairs[:, 3]-proposals_pairs[:, 1]+1)
    proposals_areas_2 = (proposals_pairs[:, 6]-proposals_pairs[:, 4]+1)*(proposals_pairs[:, 7]-proposals_pairs[:, 5]+1)

    iou = intersections_areas.float()/(proposals_areas_1+proposals_areas_2-intersections_areas)
    iou = iou.view(proposals_single.shape[0], proposals_single.shape[0])

    # compute L matrix
    L = dpp_alpha * iou * torch.ger(torch.exp(scores_single), torch.exp(scores_single))

    return torch.LongTensor(_greedy_cpp(L.cpu().numpy(), dpp_k)).cuda()        


def _greedy_cpp(L, dpp_k):
    L += np.eye(L.shape[0])*1e-8

    L = np.ascontiguousarray(L.astype('float64'))
    y = np.ascontiguousarray(np.zeros(dpp_k, dtype='int64'))

    k = _CppDPP.greedy(L, dpp_k, y)

    y = y[:k]

    return y
