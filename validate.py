import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path
from IPython.display import clear_output
import torch


def score_model(model, class_selector, metric, data,device):

    model.eval()  # testing mode
    scores = {'mean':{},'std':{}}

    ### compute sum and sum of squares in online manner

    for X_batch, Y_label in tqdm(data):

        Y_pred = class_selector(model(X_batch.to(device))) ### list of BS x 1 x H x W masks
        metric_dict = metric(Y_pred, Y_label.to(device))

        for key, val in metric_dict.items():

            if key not in scores['mean']:
                scores['mean'][key] = 0.0

            scores['mean'][key] += val

            if key not in scores['std']:
                scores['std'][key] = 0.0

            scores['std'][key] += val**2

    ### now convert sum and sum squared to mean and std

    for key in scores['mean']:
        scores['mean'][key] = scores['mean'][key]/len(data)

    for key in scores['std']:
        scores['std'][key] = scores['std'][key]/len(data) - scores['mean'][key]**2 ## variance of metric on batch
        scores['std'][key] = scores['std'][key] / len(data) ## variance of mean estimate
        scores['std'][key] = np.sqrt(scores['std'][key]) ### std

    return scores


def validate_epoch(model, loss_fn, data,device):
    
    avg_loss = 0.0

    for X_batch, Y_batch in tqdm(data):

        X_batch, Y_batch = X_batch.to(device).detach(), Y_batch.to(device).detach()
        Y_pred = model(X_batch)
        loss = loss_fn(Y_pred,Y_batch)

        avg_loss += loss.item() / len(data)

    return avg_loss


def validate(model, ckpt_path, loss_fn, class_selectors, class_content, metric, data, device):
    
    models = [p for p in Path(ckpt_path).iterdir() if p.name.endswith('.ckpt')]
    models = sorted(models, key = lambda x: int(x.name.split('.')[0].split('=')[1]))
    
    losses = []
    scores = {type(selector).__name__:{'mean':{},'std':{}} for selector in class_selectors}
    for epoch, model_path in enumerate(models):
        
        clear_output(wait = True)
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        loss = validate_epoch(model, loss_fn, data,device)
        losses.append(loss)
        
        print('epoch %d - loss: %f' % (epoch,loss))
        
        for selector in class_selectors:
            score = score_model(model, selector, metric, data,device)
            score_mean = score['mean']
            score_std = score['std']
            score_mean = {key:score_mean[key] for key in score_mean if len(key.split('_')) == 2} ## only levels data
            score_std = {key:score_std[key] for key in score_std if len(key.split('_')) == 2} ## only levels data
            
            for key in score_mean:
                if key not in scores[type(selector).__name__]['mean']:
                    scores[type(selector).__name__]['mean'][key] = []
                    scores[type(selector).__name__]['std'][key] = []
                scores[type(selector).__name__]['mean'][key].append(score_mean[key])
                scores[type(selector).__name__]['std'][key].append(score_std[key])
    
    return losses, scores

