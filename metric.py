import numpy as np
import torch
from .hierarchical_utils import decompose_mask

class HierarchicalMIoU:

    def __init__(self, class_content, smooth = 1e-8):
        self.class_content = class_content
        self.smooth = smooth
        self.n_levels = len(class_content)

    def iou(self, outputs : np.array, labels: np.array):

        #inputs are boolean BS x W x H arrays

        assert(isinstance(outputs,np.ndarray))
        assert(isinstance(labels,np.ndarray))
        assert(outputs.ndim == 3)
        assert(labels.ndim == 3)
        assert(outputs.dtype == np.bool_)
        assert(labels.dtype == np.bool_)

        num = (outputs & labels).sum((1,2)) ### sum over BS x W x H
        denum = (outputs | labels).sum((1,2)) ### sum over BS x W x H
        metric = num / (denum + self.smooth) ### smooth for numerical stability
        metric = metric.mean() ### mean value over batch
        
        return metric.item()

    def __call__(self, outputs: list, labels: torch.Tensor):

        ### outputs is list of N_LEVELS tensors of shape BS x H x W
        ### labels is tensor of shape BS x H x W

        assert (len(outputs) == self.n_levels)
        labels = decompose_mask(labels,self.class_content)
        ### now labels is list of  N_LEVELS tensors of shape BS x H x W

        error = {}

        for i in range(self.n_levels):
            level = f'level_{i+1}'
            error[level] = 0.0
            
            for j,key in enumerate(self.class_content[level]):
                classes = self.class_content[level][key]
                class_ = j+1 if len(classes) > 1 else classes[0] ### on last level get original label
                metric = self.iou((outputs[i].to('cpu').numpy()==class_),
                                  (labels[i].to('cpu').numpy() == class_))
                error[level+'_'+key] = metric
                error[level] += metric
            error[level] /= len(self.class_content[level])

        return error
