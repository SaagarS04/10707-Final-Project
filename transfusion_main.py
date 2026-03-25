import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import os
import json
import pathlib
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import seaborn as sb

from torch.utils.tensorboard import SummaryWriter

import argparse

from Transfusion import TransEncoder, GaussianDiffusion1D
from ddpm import *
from data_make import *

from pitch_data import get_transformed_data

import warnings
warnings.filterwarnings('ignore')


def visualize(ori_data, fake_data, dataset_name, seq_len, save_path, epoch, writer):
    
    ori_data = np.asarray(ori_data)

    fake_data = np.asarray(fake_data)
    
    ori_data = ori_data[:fake_data.shape[0]]
    
    sample_size = 250
    
    idx = np.random.permutation(len(ori_data))[:sample_size]
    
    randn_num = np.random.permutation(sample_size)[:1]
    
    real_sample = ori_data[idx]

    fake_sample = fake_data[idx]
    
        
    real_sample_2d = real_sample.reshape(-1, seq_len)

    fake_sample_2d = fake_sample.reshape(-1, seq_len)
    
    mode = 'visualization'
        
    ### PCA
    
    pca = PCA(n_components=2)
    pca.fit(real_sample_2d)
    pca_real = (pd.DataFrame(pca.transform(real_sample_2d))
                .assign(Data='Real'))
    pca_synthetic = (pd.DataFrame(pca.transform(fake_sample_2d))
                     .assign(Data='Synthetic'))
    pca_result = pd.concat([pca_real, pca_synthetic], ignore_index=True).rename(
        columns={0: '1st Component', 1: '2nd Component'})
    
    
    ### TSNE
    
    tsne_data = np.concatenate((real_sample_2d,
                            fake_sample_2d), axis=0)

    tsne = TSNE(n_components=2,
                verbose=0,
                perplexity=40)
    tsne_result = tsne.fit_transform(tsne_data)
    
    
    tsne_result = pd.DataFrame(tsne_result, columns=['X', 'Y']).assign(Data='Real')
    
    tsne_result.loc[len(real_sample_2d):, 'Data'] = 'Synthetic'
    
    fig, axs = plt.subplots(ncols = 2, nrows=2, figsize=(20, 20))

    sb.scatterplot(x='1st Component', y='2nd Component', data=pca_result,
                    hue='Data', style='Data', ax=axs[0,0])
    sb.despine()
    
    axs[0,0].set_title('PCA Result')


    sb.scatterplot(x='X', y='Y',
                    data=tsne_result,
                    hue='Data', 
                    style='Data', 
                    ax=axs[0,1])
    sb.despine()

    axs[0,1].set_title('t-SNE Result')

    axs[1,0].plot(real_sample[randn_num[0], :, :])

    axs[1,0].set_title('Original Data')

    axs[1,1].plot(fake_sample[randn_num[0], :, :])

    axs[1,1].set_title('Synthetic Data')

    fig.suptitle('Assessing Diversity: Qualitative Comparison of Real and Synthetic Data Distributions', 
                 fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=.88)
    
    plt.savefig(os.path.join(f'{save_path}', f'{time.time()}-tsne-result-{dataset_name}.png'))
     
    writer.add_figure(mode, fig, epoch)


def main(args):
    
    seq_len = args.seq_len
    epochs = args.training_epoch
    timesteps = args.timesteps
    batch_size = args.batch_size
    latent_dim = args.hidden_dim
    num_layers = args.num_of_layers
    n_heads = args.n_head    
    dataset_name = args.dataset_name
    beta_schedule = args.beta_schedule
    objective = args.objective
    
    train_dict = get_transformed_data('2024-05-12', '2025-10-01', weather=False)
    test_dict = get_transformed_data('2025-10-01', '2025-11-03', weather=False)

    print("Processing datasets into sequences...")
    def process_dict(d):
        pc = d['pitch_context'].drop(columns=['age_pit', 'age_bat', 'if_fielding_alignment', 'of_fielding_alignment', 'oaa'], errors='ignore')
        pc[['on_1b', 'on_2b', 'on_3b']] = pc[['on_1b', 'on_2b', 'on_3b']].fillna(0)
        pitch = d['pitch'].copy()
        return pd.concat([pc, pitch], axis=1)

    train_combined = process_dict(train_dict)
    test_combined = process_dict(test_dict)
    
    train_combined['is_train'] = 1
    test_combined['is_train'] = 0
    all_combined = pd.concat([train_combined, test_combined])
    
    # One-hot encode categoricals (like pitch types and defensive formations)
    all_combined = pd.get_dummies(all_combined, columns=all_combined.select_dtypes(include=['object']).columns, dummy_na=False, drop_first=True).astype(float)
    
    train_df = all_combined[all_combined['is_train'] == 1].drop(columns=['is_train'])
    test_df = all_combined[all_combined['is_train'] == 0].drop(columns=['is_train'])
    
    # Fill remaining NaNs with train means
    mean_vals = train_df.mean().fillna(0)
    train_df = train_df.fillna(mean_vals)
    test_df = test_df.fillna(mean_vals)
    
    train_arr = train_df.values
    test_arr = test_df.values
    
    # Truncate and reshape to (batch, seq_len, features)
    n_train = (train_arr.shape[0] // seq_len) * seq_len
    train_data = train_arr[:n_train].reshape(-1, seq_len, train_arr.shape[1])
    
    n_test = (test_arr.shape[0] // seq_len) * seq_len
    test_data = test_arr[:n_test].reshape(-1, seq_len, test_arr.shape[1])

    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)
    
    features = train_data.shape[2]
    
    # Transpose to (batch, features, seq_len) as expected by PyTorch Conv1d
    train_data, test_data = train_data.transpose(0,2,1), test_data.transpose(0,2,1)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size)
    
    test_loader = torch.utils.data.DataLoader(test_data, len(test_data))
    
    real_data = next(iter(test_loader))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    mode = 'diffusion'
    
    architecture = 'custom-transformers'
    
    loss_mode = 'l1'
    
    file_name = f'{architecture}-{dataset_name}-{loss_mode}-{beta_schedule}-{seq_len}-{objective}'
    
    folder_name = f'saved_files/{time.time():.4f}-{file_name}'
    
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True) 
    
    gan_fig_dir_path = f'{folder_name}/output/gan'
    
    pathlib.Path(gan_fig_dir_path).mkdir(parents=True, exist_ok=True)
    
    file_name_gan_fig = f'{file_name}-gan'
    
    with open(f'{folder_name}/params.txt', 'w') as f:
        
        json.dump(args.__dict__, f, indent=2)
        
        f.close() 
    
    writer = SummaryWriter(log_dir = folder_name, comment = f'{file_name}', flush_secs = 45)
    
    
    model = TransEncoder(
    
        features = features,
        latent_dim = latent_dim,
        num_heads = n_heads,
        num_layers = num_layers
    
    )

    diffusion = GaussianDiffusion1D(
        model,
        seq_length = seq_len,
        timesteps = timesteps,  
        objective = objective, # pred_x0, pred_v
        loss_type = 'l2',
        beta_schedule = beta_schedule
    )
    
    diffusion = diffusion.to(device)

    lr = 1e-4
    
    betas = (0.9, 0.99)

    optim = torch.optim.Adam(diffusion.parameters(), lr = lr, betas = betas)
    
    
    for running_epoch in tqdm(range(epochs)):
        
        for i, data in enumerate(train_loader):
            
            data = data.to(device)
            
            batch_size = data.shape[0]
            
            optim.zero_grad()
            
            loss = diffusion(data)
            
            loss.backward()
            
            optim.step()
            
            if i%len(train_loader)==0:
                
                writer.add_scalar('Loss', loss.item(), running_epoch)
                
            if i%len(train_loader)==0 and running_epoch%100==0:
                
                print(f'Epoch: {running_epoch+1}, Loss: {loss.item()}')
                
            if i%len(train_loader)==0 and running_epoch%500==0:
                
                with torch.no_grad():
                    
                    samples = diffusion.sample(len(test_data))

                    samples = samples.cpu().numpy()

                    samples = samples.transpose(0, 2, 1)
                    
                    np.save(f'{folder_name}/synth-{dataset_name}-{seq_len}-{running_epoch}.npy', samples)
                    
                visualize(real_data.cpu().numpy().transpose(0,2,1), samples, dataset_name, seq_len, gan_fig_dir_path, running_epoch, writer)
                
                
    torch.save({

        'epoch': running_epoch+1,
        'diffusion_state_dict': diffusion.state_dict(),
        'diffusion_optim_state_dict': optim.state_dict()

        }, os.path.join(f'{folder_name}', f'{file_name}-final.pth'))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--dataset_name',
        choices=['sine','stock','air', 'energy'],
        default='sine',
        type=str)
    
    parser.add_argument(
        '--beta_schedule',
        choices=['cosine','linear', 'quadratic', 'sigmoid'],
        default='cosine',
        type=str)
    
    parser.add_argument(
        '--objective',
        choices=['pred_x0','pred_v', 'pred_noise'],
        default='pred_v',
        type=str)
    
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=100,
        type=int)
    
    parser.add_argument(
        '--batch_size',
        help='batch size for the network',
        default=256,
        type=int)
    
    parser.add_argument(
        '--n_head',
        help='number of heads for the attention',
        default=8,
        type=int)
    
    parser.add_argument(
        '--hidden_dim',
        help='number of hidden state',
        default=256,
        type=int)
    
    parser.add_argument(
        '--num_of_layers',
        help='Number of Layers',
        default=6,
        type=int)
    
    parser.add_argument(
        '--training_epoch',
        help='Diffusion Training Epoch',
        default=5000,
        type=int)
    
    parser.add_argument(
        '--timesteps',
        help='Timesteps for Diffusion',
        default=1000,
        type=int)
    
    args = parser.parse_args() 
    
    main(args)