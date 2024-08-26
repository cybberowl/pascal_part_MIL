import numpy as np
from tqdm.notebook import tqdm

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