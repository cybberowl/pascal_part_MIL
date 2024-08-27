from  datetime import datetime
import pandas as pd
from tqdm.notebook import tqdm
from IPython.display import clear_output
import yaml
import torch
from .plotting import plot_examples_learning
import numpy as np

dt_format = '%Y-%m-%d %H:%M'

def get_experiment_params(model, opt, loss_fn, data):
    
    types = [list, int, bool,float, str]

    config = {}
    config['model_params'] = model.config
    config['lr'] = opt.param_groups[0]['lr']
    config['batch_size'] = data.batch_size
    config['resample'] = data.dataset.resample
    config['augmentations'] = [type(t).__name__ for t in data.dataset.transform] if data.dataset.transform else None
    config['loss'] = type(loss_fn).__name__
    if hasattr(loss_fn, 'base_loss_class'):
        config['base_loss'] = loss_fn.base_loss_class.__name__

    return config

def train_one_epoch(model, opt, loss_fn, data, device):

    avg_loss = 0.0

    for X_batch, Y_batch in tqdm(data):

        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        opt.zero_grad()

        Y_pred = model(X_batch)
        loss = loss_fn(Y_pred,Y_batch)
        loss.backward()

        opt.step()

        avg_loss += loss.item() / len(data)

    return avg_loss

def train(model, opt, loss_fn, class_selector, class_content, epochs, data, data_val, experiment_name,ckpt_path, device):

    train_loss = []

    ## dump config
    config = get_experiment_params(model, opt, loss_fn, data)
    date = datetime.now().strftime(dt_format)
    path = ckpt_path / f"{experiment_name} {date}"
    path.mkdir(exist_ok = True, parents = True)
    
    with open(path/'config.yaml','w') as f:
        yaml.safe_dump(config, f)

    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        model.train()  # train mode

        avg_loss = train_one_epoch(model, opt, loss_fn, data, device)
        
        print('loss: %f' % avg_loss)
        train_loss.append(avg_loss)

        ## model save
        model_path  = path / f'epoch={epoch+1}.ckpt'
        torch.save(model.state_dict(), model_path)

        model.eval() # testing mode

        X_val, Y_val = next(iter(data_val))
        X_val, Y_val = X_val.to(device).detach(), Y_val.to(device).detach()
        Y_hat = class_selector(model(X_val))

        clear_output(wait=True)
        plot_examples_learning(X_val,Y_val, Y_hat, decomposed=True, class_content = class_content)

        print('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))


        # avg_loss = 0.0
        # for X_batch, Y_batch in data_val:
        #     # data to device
        #     X_batch, Y_batch = X_batch.to(device).detach(), Y_batch.to(device).detach()
        #     Y_pred = model(X_batch)
        #     loss = loss_fn(Y_pred,Y_batch)
        #     avg_loss += loss.detach().to('cpu').numpy() / len(data_val)
        # val_loss.append(avg_loss)

    pd.Series(train_loss, index = np.arange(1,epochs+1)).to_csv(path/'train_loss.csv')

    return train_loss, path
