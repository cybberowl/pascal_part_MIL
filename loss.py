import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .hierarchical_utils import decompose_mask, aggregate_probs

def compute_weights(data, class_content, bg_class = 0):

    n_levels = len(class_content)
    n_classes = len(class_content[f'level_{n_levels}']) + 1

    freqs = {i:0 for i in range(n_classes)}

    for x_batch, y_batch in data:
        for i in range(n_classes):
            freqs[i] += (y_batch.numpy() == i).sum()
    
    sum_ = sum(freqs.values())
    freqs = {k:v/sum_ for k,v in freqs.items()}

    res = {}

    for i in range(1,n_levels+1):
        res[f'level_{i}'] = {}
        counter = 0
        res[f'level_{i}'][counter] = 1/freqs[bg_class] ## bg_class
        for key, classes in class_content[f'level_{i}'].items():
            counter += 1
            cnt = counter if len(classes) > 1 else classes[0]
            res[f'level_{i}'][cnt] = 1/sum(freqs[c] for c in classes)  
        res[f'level_{i}'] = np.array([res[f'level_{i}'][c] for c in range(len(class_content[f'level_{i}'])+1)])
        res[f'level_{i}'] = res[f'level_{i}']/res[f'level_{i}'].sum() 
        res[f'level_{i}'] = torch.Tensor(res[f'level_{i}']) 

    return list(res.values())

class HierarchicalLoss:

    def __init__(self, base_loss_class, class_weights, class_content, smooth = 1e-6, **kwargs):

        self.base_loss_class = base_loss_class
        self.class_content = class_content
        self.smooth = smooth
        self.losses = [base_loss_class(weight = w, **kwargs) for w in class_weights]
    
    def __call__(self, Y_pred,Y_true):

        loss = 0
        true_masks = decompose_mask(Y_true, self.class_content)
        probs = F.softmax(Y_pred, dim = 1)
        probs = aggregate_probs(probs,self.class_content)
        logits = [torch.log(p+self.smooth) for p in probs]

        for logit, mask, loss_func in zip(logits, true_masks, self.losses):
            loss += loss_func(logit, mask)
        
        return loss

class DiceLoss:

    def __init__(self, weight, size, smooth = 1e-6):

        self.size = size
        self.smooth = smooth
        self.weight = weight 

    def __call__(self,Y_pred,Y_true):

        pred_mask = F.softmax(Y_pred, dim = 1)
        n_classes = Y_pred.shape[1]
        res = 0.0

        for i in range(1, n_classes): ## 0 is bg class
            
            y_true = (Y_true == i).int()
            num = 2* (y_true*pred_mask[:,i,...]).sum((1,2)) ## sum over WxH dimensions
            den = y_true.sum((1,2)) + pred_mask[:,i,...].sum((1,2))
            value = 1 - ((num + self.smooth) / (den + self.smooth )) / self.size[0]*self.size[1]
            if self.weight is not None:
                value = value*self.weight[i]
            res += value.mean()
        
        return res

class FocalLoss:

    def __init__(self, weight, gamma = 2, smooth = 1e-6):

        self.gamma = gamma
        self.smooth = smooth
        self.weight = weight 

    def __call__(self, Y_pred,Y_true):

        probs = torch.clamp(F.softmax(Y_pred, dim = 1), self.smooth, 1.0-self.smooth)
        n_classes = Y_pred.shape[1]
        res = 0.0

        for i in range(n_classes): ## 0 is bg class
            
            y_true = (Y_true == i).int()
            value = - torch.pow(1 - probs[:,i,...], self.gamma) * y_true * torch.log(probs[:,i,...])
            if self.weight is not None:
                value = value*self.weight[i]
            res += value.mean()
        
        return res
