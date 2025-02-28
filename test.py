import numpy as np
import pandas as pd
import json 
from time import time
import argparse

import os
import sys
sys.path.append('./models')

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from models.model import MixForecast

from tqdm import tqdm
from time import time
# from sklearn.metrics import mean_squared_error

# metrics used for evaluation
def cal_cvrmse(pred, true, eps=1e-8):
    pred = np.array(pred)
    true = np.array(true)
    return np.power(np.square(pred - true).sum() / pred.shape[0], 0.5) / (true.sum() / pred.shape[0] + eps)

def cal_mae(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    return np.mean(np.abs(pred - true))

def cal_nrmse(pred, true, eps=1e-8):
    true = np.array(true)
    pred = np.array(pred)

    M = len(true) // 24
    y_bar = np.mean(true)
    NRMSE = 100 * (1/ (y_bar+eps)) * np.sqrt((1 / (24 * M)) * np.sum((true - pred) ** 2))
    return NRMSE


def standardize_series(series, eps=1e-8):
    mean = np.mean(series)
    std = np.std(series)
    standardized_series = (series - mean) / (std+eps)
    return standardized_series, mean, std

def unscale_predictions(predictions, mean, std, eps=1e-8):
    return predictions * (std+eps) + mean


class TimeSeriesDataset(Dataset):
    def __init__(self, data, backcast_length, forecast_length, stride=1):
        # Standardize the time series data
        self.data, self.mean, self.std = standardize_series(data)
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.stride = stride

    def __len__(self):
        return (len(self.data) - self.backcast_length - self.forecast_length) // self.stride + 1

    def __getitem__(self, index):
        start_index = index * self.stride
        x = self.data[start_index : start_index + self.backcast_length]
        y = self.data[start_index + self.backcast_length : start_index + self.backcast_length + self.forecast_length]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), torch.tensor(y, dtype=torch.float32).unsqueeze(0)


def test(args, model, criterion, device):

    folder_path = os.path.join(args["dataset_path"], "test")
    result_path = args["result_path"]
    
    median_res = []  
    for region in os.listdir(folder_path):

        region_path = os.path.join(folder_path, region)

        results_path = os.path.join(result_path, region)
        os.makedirs(results_path, exist_ok=True)

        res = []

        for building in os.listdir(region_path):

            building_id = building.rsplit(".csv",1)[0]

            if building.endswith('.csv'):
                file_path = os.path.join(region_path, building)
                df = pd.read_csv(file_path)
                energy_data = df['energy'].values
                dataset = TimeSeriesDataset(energy_data, args["seq_len"], args["pred_len"], args["stride"])
                
                # test phase
                model.eval()
                val_losses = []
                y_true_test = []
                y_pred_test = []

                # test loop
                for x_test, y_test in tqdm(DataLoader(dataset, batch_size=1), desc=f"Testing {building_id}", leave=False):
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    with torch.no_grad():
                        backcast, forecast = model(x_test)
                        loss = criterion(forecast, y_test)
                        val_losses.append(loss.item())
                        
                        # Collect true and predicted values for RMSE calculation   
                        y_test = y_test.squeeze(1)
                        forecast = forecast.squeeze(1)
                        y_true_test.extend(y_test.cpu().numpy())
                        y_pred_test.extend(forecast.cpu().numpy())
                        
                # Calculate average validation loss and RMSE
                y_true_combine = np.concatenate(y_true_test, axis=0)
                y_pred_combine = np.concatenate(y_pred_test, axis=0)
                avg_test_loss = np.mean(val_losses)
                
                y_pred_combine_unscaled = unscale_predictions(y_pred_combine, dataset.mean, dataset.std)
                y_true_combine_unscaled = unscale_predictions(y_true_combine, dataset.mean, dataset.std)
                
                # Calculate CVRMSE, NRMSE, MAE on unscaled data
                cvrmse = cal_cvrmse(y_pred_combine_unscaled, y_true_combine_unscaled)
                nrmse = cal_nrmse(y_pred_combine_unscaled, y_true_combine_unscaled)
                mae = cal_mae(y_pred_combine_unscaled, y_true_combine_unscaled)

                res.append([building_id, cvrmse, nrmse, mae, avg_test_loss])

        columns = ['building_ID', 'CVRMSE', 'NRMSE', 'MAE', 'Avg_Test_Loss']
        df = pd.DataFrame(res, columns=columns)
        df.to_csv("{}/{}.csv".format(results_path, 'result'), index=False)

        med_nrmse = df['NRMSE'].median()
        median_res.append([region, med_nrmse])

    med_columns = ['Dataset','NRMSE']
    median_df = pd.DataFrame(median_res, columns=med_columns)
    median_df.to_csv(f"{result_path}/median_buildings_results.csv", index=False)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    parser.add_argument('--config-file', type=str, default='./configs/model_base.json', help='Input config file path', required=True)
    file_path_arg = parser.parse_args()
    config_file = file_path_arg.config_file
    # config_file = './configs/model_base.json'
    with open(config_file, 'r') as f:
        args = json.load(f)


    # check device 
    device = args["device"]
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    num_patches = args["seq_len"] // args["patch_size"]
    
    # Define MixForecast model
    model = MixForecast(
        device=device,
        forecast_length=args["pred_len"],
        backcast_length=args["seq_len"],
        patch_size = args["patch_size"], 
        num_patches = num_patches,
        num_features = args["num_features"],
        hidden_dim=args["hidden_dim"],
        nb_blocks_per_stack=args["num_blocks_per_stack"],
        stack_layers = args["stack_layers"],
        factor = args["factor"],
    ).to(device)

    model.load_state_dict(torch.load(f'{args["model_save_path"]}/best_model.pth'))

    # model's parameters
    param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model's parameter count is:", param)

    # Define loss
    if args["loss"] == "huber":
        criterion = torch.nn.HuberLoss(reduction="mean", delta=1.0)
    else:
        criterion = torch.nn.MSELoss()

    start_time = time()

    # testing the model
    test(args=args, model=model, criterion=criterion, device=device)


    end_time = time() - start_time

    print(f"inference time taken by model is {end_time} sec")


