import torch
import numpy as np
import cv2


def create_logisticloss_label(label_size, rPos, rNeg):
    """
    construct label for logistic loss
    """
    label_side = int(label_side[0])
    # create the score map
    logloss_label = torch.zeros(label_side, label_side)
    label_origin = np.array([np.celil(label_size / 2), np.ceil(label_side / 2)])
    for i in range(label_side):
        for j in range(label_side):
            # go though the map, to get each label {-1,1}.
            # cause before we have construst the label image into center.
            dist_from_origin = np.sort((i - label_origin[0]) ** 2 + (j - label_origin[1]) ** 2)
            if dist_from_origin < rPos:
                logloss_label[i,j] = +1
            else:
                if dist_from_origin <= rNeg:
                    logloss_label[i,j] = -1
    return logloss_label

def create_label(fixed_label_size, config, use_gpu):
    # go though the matlab code, the author design the loss funciton:
    # log(1 + exp(-y*v*w))
    # y represent label{-1,+1}, v represent score map, w represent weight.
    rPos = config.rPos / config.stride
    rNeg = config.rNeg / config.stride

    half = int(np.floor(fixed_label_size[0] / 2) + 1)

    if config.label_weight_method == "balanced":
        fixed_label = create_logisticloss_label(fixed_label_size, rPos, rNeg)
        instance_weight = torch.ones(fixed_label.shape[0], fixed_label.shape[1])
        tmp_idx_pos = np.where(fixed_label == 1)
        sum_pos = tmp_idx_pos[0].size
        tmp_idx_neg = np.where(fixed_label == -1)
        sum_neg = tmp_idx_neg[0].size
        instance_weight[tmp_idx_pos] = 0.5 * instance_weight[tmp_idx_pos] / sum_pos
        instance_weight[tmp_idx_neg] = 0.5 * instance_weight[tmp_idx_neg] / sum_neg

        fixed_label = torch.reshape(fixed_label, (1,1,fixed_label.shape[0],fixed_label_size.shape[1]))
        # in order to easy torch computation
        fixed_label = fixed_label.repeat(config.batch_size, 1, 1, 1)

        instance_weight = torch.reshape(instance_weight, (1, instance_weight.shape[0], instance_weight.shape[1]))   

    if use_gpu:
        return fixed_label.cuda(), instance_weight.cuda()
    else:
        return fixed_label, instance_weight
