import torch
import torch.nn.functional as F
from .hierarchical_utils import decompose_mask,aggregate_probs

class SimpleHierarchicalClassSelector:

    def __init__(self, class_content):

        self.class_content = class_content

    def __call__(self, x_batch):

        ### x_batch is BS x C x H x W

        prob = F.softmax(x_batch.detach(), dim = 1) # BS x C x H x W -> BS x C x H x W
        labels = torch.argmax(prob, dim = 1) # BS x C x H x W -> BS x H x W

        res = decompose_mask(labels, self.class_content) # BS x H x W  -> list of BS x H x W
        return res

class SmartHierarchicalClassSelector:

    def __init__(self, class_content):

        self.class_content = class_content

    def __call__(self, x_batch):

        ### x_batch is BS x C x H x W

        prob = F.softmax(x_batch.detach(), dim = 1)  # BS x C x H x W -> BS x C x H x W
        prob_array = aggregate_probs(prob,self.class_content) # BS x C x H x W -> list of N_LEVELS tensors BS x C x H x W
        labels = [torch.argmax(p,dim = 1) for p in prob_array]

        return labels
