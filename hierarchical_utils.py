import torch

# +
# def device_converter(tensor):

#     return 'cpu' if tensor.get_device()<0 else 'cuda'
# -

def decompose_mask(mask:torch.Tensor,class_content):
    '''
    convert one mask to N_LEVELS masks
    mask in tensor BS x H x W
    '''
    assert(mask.ndim == 3)
    res = []
    device = mask.device
    for level in sorted(class_content,key = lambda x: int(x.split('_')[1])):
        new_mask = mask.clone().to(device) ### zeros remain zeros
        counter = 1
        for key, classes in class_content[level].items():
            ### counter is label for new class on mid levels
            ### for lowest level (classes is singleton list) we use original labels
            ### bg class always remains 0 on every level
            new_mask[torch.isin(mask,torch.Tensor(classes).to(device))] = counter if len(classes) > 1 else classes[0]
            counter += 1
        res.append(new_mask)
    return res

def aggregate_probs(probs: torch.Tensor, class_content, bg_class = 0):
    '''
    aggregate low-level probabilities to N_LEVELS probabilities of higher classes
    probs is tensor BS x C x H x W
    '''
    
    device = probs.device
    res = []

    for level in sorted(class_content,key = lambda x: int(x.split('_')[1])):

        shape = list(probs.shape)
        shape[1] = len(class_content[level]) + 1 ### extra size for bg class
        new_probs = torch.zeros(size = shape, dtype = torch.float32).to(device)
        idx = 0
        new_probs[:,idx,...] = probs[:,bg_class,...] ### here we use it
        for key, classes in class_content[level].items():
            idx += 1
            for c in classes:
                new_probs[:,idx,...] += probs[:,c,...]
        res.append(new_probs)
    return res

