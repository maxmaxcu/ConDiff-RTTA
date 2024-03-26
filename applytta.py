import hydra
# from hydra.utils import get_original_cwd
# from omegaconf import OmegaConf, open_dict
# from mergedeep import merge
import numpy as np
import pickle
import torch
# import torch.backends.cudnn as cudnn
torch.backends.cudnn.benchmark = True
import dnnlib
import copy

import os
import warnings
warnings.filterwarnings("ignore")
from tta.models_tta import TTAGradientDescent_Class
from tta.models_discr import TCNet
from tta.utils import tta_one_image_by_gradient_descent
from tta.dataset import TCIRDataset_TTA

def load_pretrained_models_tta(config,diff_path=None, discr_path=None, freeze_diff=True):
    # Load Diffusion Networks
    device = "cuda" if config.gpu is None else "cuda:{}".format(config.gpu)
    
    if diff_path == None:
        print('No Pretrained Diff Model')
        model_diff = None
    else:
        with dnnlib.util.open_url(diff_path, verbose=True) as f:
            model_diff = pickle.load(f)['ema'].to(device)
            print('Diffusion Model Loaded',diff_path)
        model_diff.eval()
        if freeze_diff:
            for param in model_diff.parameters():
                param.requires_grad = False
    
    # Load Discriminative Networks
    if discr_path == None:
        print('No Pretrained Discr Model')
        model_discr = None
    else:
        model_discr = TCNet(name='resnet18',inchannel=1).to(device)
        model_discr.load_state_dict(
            torch.load(
                discr_path,
                map_location=device,
            )
        )
        model_discr.eval()
        print('Pretrained Discriminative Model Loaded!')
    # return torch.compile(model_diff), torch.compile(model_discr)
    return model_diff, model_discr


def tta_one_epoch(config, dataloader, tta_model, optimizer, scaler):
    device = "cuda" if config.gpu is None else "cuda:{}".format(config.gpu)
    tta_model.eval()
    tta_class_state_dict = copy.deepcopy(tta_model.state_dict())
    before_errors = []
    after_errors = []
    before_preds = []
    after_preds = []
    gts = []
    sum_impr = 0

    # Start iterations
    for idx, batch in enumerate(dataloader):

        # Fetch data from the dataset
        print(f"\n\n Example: {idx}/{len(dataloader)} \n\n")
        data_x_discr, data_x_diff, data_y = batch
        data_x_discr = data_x_discr.to(device)
        data_x_diff = data_x_diff.to(device)
        gts.append(data_y.numpy())
        data_y = data_y.to(device)
        batch = (data_x_discr, data_x_diff, data_y)

        # Step 1: Predict pre-TTA classification. The results are saved in
        before_tta_stats_dict = tta_model.evaluate(data_x_discr, data_y, before_tta=True, bs=config.input.batch_size, config=config)
        before_pred = before_tta_stats_dict['before_tta_preds']
        before_error = torch.abs(before_pred - data_y).item()
        before_pred = before_pred.item()
        before_errors.append(before_error)
        before_preds.append(before_pred)

        # Step 2: TTA by gradient descent
        if config.tta.model.method == 'tta':
            losses = tta_one_image_by_gradient_descent(config, batch, tta_model, optimizer, scaler)
            
        # Step 3: Predict post-TTA classification. The results are saved in
        if config.tta.model.method == 'tta':
            after_tta_stats_dict = tta_model.evaluate(data_x_discr, data_y, after_tta=True, bs=config.input.batch_size, config=config)
            after_pred = after_tta_stats_dict['after_tta_preds']

        if config.input.lower_bound != None and after_pred < config.input.lower_bound:
            if config.tta.model.method == 'tta':
                after_pred = torch.tensor(config.input.lower_bound)
        if config.input.upper_bound != None and after_pred > config.input.upper_bound:
            if config.tta.model.method == 'tta':
                after_pred = torch.tensor(config.input.upper_bound)
        
        after_error = torch.abs(after_pred - data_y).item()
        after_pred = after_pred.item()
        after_errors.append(after_error)
        after_preds.append(after_pred)

        improvement = before_error - after_error
        sum_impr += improvement
        print(f"GT: {data_y.item():.3f}, Before: {before_pred:.3f}, After: {after_pred:.3f}")
        print(f"Before error: {before_error:.3f}, After error: {after_error:.3f}, Improvement: {improvement:.3f}")
        print(f"Avg improvement so far: {sum_impr / (idx + 1):.3f}")
        
        # Reload the original model state dict
        if not config.tta.model.online:
            tta_model.load_state_dict(tta_class_state_dict)
            optimizer = torch.optim.AdamW(
                tta_model.parameters(), lr=config.tta.gradient_descent.base_learning_rate,
                weight_decay=0,
            )
            optimizer.zero_grad()

        if config.input.testdatasize!= 0 and idx >=  config.input.testdatasize-1:
            break
        
    before_errors = np.array(before_errors)
    before_mae = before_errors.mean()
    before_rmse = (before_errors**2).mean()**0.5
    after_errors = np.array(after_errors)
    after_mae = after_errors.mean()
    after_rmse = (after_errors**2).mean()**0.5
    before_preds = np.array(before_preds).squeeze()
    after_preds = np.array(after_preds).squeeze()
        
    gts = np.array(gts).squeeze()
    if not os.path.exists(f'{os.getcwd()}/results'):
        os.makedirs(f'{os.getcwd()}/results')
    np.save(f'{os.getcwd()}/results/before_preds.npy', before_preds)
    np.save(f'{os.getcwd()}/results/after_preds.npy', after_preds)
    np.save(f'{os.getcwd()}/results/gts.npy', gts)
    print(f"Before MAE: {before_mae:.2f}, After MAE: {after_mae:.2f}, Improvement: {(before_mae - after_mae):.2f}")
    print(f"Before RMSE: {before_rmse:.2f}, After RMSE: {after_rmse:.2f}, Improvement: {(before_rmse - after_rmse):.2f}")


@hydra.main(config_path="tta", config_name="config")
def main(config):
    device = "cuda" if config.gpu is None else "cuda:{}".format(config.gpu)
    dataset = TCIRDataset_TTA(path=config.input.dataset_path,phase=config.input.dataset_phase, config=config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=config.input.dataset_shuffle,
        num_workers=1,
        pin_memory=True,
    )
    print(f'phase:{config.input.dataset_phase}')
    
    model_diff_tta, model_discr_tta = load_pretrained_models_tta(config, diff_path=config.input.diff_model, discr_path=config.input.discr_model,freeze_diff=config.input.freeze_diff)
    

    model_tta = TTAGradientDescent_Class(config, model_diff_tta, model_discr_tta, dataloader.batch_size).to(device)

    optimizer = torch.optim.AdamW(
            model_tta.parameters(), lr=config.tta.gradient_descent.base_learning_rate,
            weight_decay=0,
        )
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler()
    tta_one_epoch(config, dataloader, model_tta, optimizer, scaler)




if __name__ == '__main__':
    main()