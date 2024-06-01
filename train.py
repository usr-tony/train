import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import pandas as pd
from pathlib import Path
from cachetools import cached
from tab_transformer_pytorch import FTTransformer
from numerapi import NumerAPI
from numerai_tools.scoring import numerai_corr
import numpy as np

pd.options.display.float_format = lambda x: f'{x:.5f}'
DATA_VERSION = 'v4.3'
numerai_data_path = Path(DATA_VERSION)
EMBED_DIM = 64
BATCH_SIZE = 2 ** 12
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device=}')


def main():
    download_datasets()
    model = FTTransformer(
        categories=[],
        num_continuous=len(get_features()),
        dim=EMBED_DIM,
        depth=6,
        heads=8,
        attn_dropout=0.2,
        ff_dropout=0.1,
    )
    loss_func = nn.L1Loss() 
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=1e-6,
        nesterov=True,
        momentum=0.6,
    )
    train_df = get_train_df(get_features())
    train = RandomData(train_df)
    validation_df = get_validation_df(get_features())
    train_loader = DataLoader(train, batch_size=BATCH_SIZE)
    model.to(device)
    running_loss = 0
    best_era_corr = 0
    for epoch in range(10):
        model.train()
        for i, (x, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            y = model(torch.Tensor([]), x)
            loss = loss_func(y, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 9 == 0:
                progress = i * BATCH_SIZE / len(train) * 100 
                print(f'{epoch=} {progress:.2f}% {running_loss=:.5f}')
                running_loss = 0

        era_corrs = validate(model, validation_df)
        total_corr = era_corrs['prediction'].sum()
        print(f'{total_corr=} {best_era_corr=}')
        if best_era_corr < total_corr:
            best_era_corr = total_corr
            torch.save(model, f'model_epoch_{epoch}.pkl')


class Embedding(nn.Module):
    """the saint embedding module is creates a feedforward layer for each feature with relu activation"""
    def __init__(self, nfeatures, embed_dim=EMBED_DIM):
        super().__init__()
        get_params = lambda: nn.Parameter(torch.rand(nfeatures, embed_dim))
        self.weights = get_params()
        self.biases = get_params()

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases


class Transformer(nn.Module):
    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        inner_dim = dim * 4
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(), # GEGLU is used in the paper
            nn.Dropout(0.8),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x):
        attn_in = self.norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = attn_out + x # should the residual be applied to before or after the norm?
        return self.ff(x) + x # residual


class Model(nn.Module):
    def __init__(self, nfeatures, embed_dim=EMBED_DIM):
        super().__init__()
        self.nfeatures = nfeatures
        
        self.embedding = Embedding(nfeatures)
        self.transformer = Transformer(embed_dim)
        self.inter_sample_transformer = Transformer(embed_dim * nfeatures)
        self.final = nn.Sequential(
            # layernorm here?
            nn.Linear(EMBED_DIM, 999),
            nn.ReLU(),
            nn.Linear(999, 1),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = rearrange(x, 'b n d -> 1 b (n d)')
        x = self.inter_sample_transformer(x)
        x = rearrange(x, "1 b (n d) -> b n d", n=self.nfeatures)

        return self.final(x[:, 0]) # apply only the class tokens/first feature


class DataByEra(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.era_counts = df['era'].value_counts()
        self.df = df.set_index('era')

    def __len__(self):
        return len(self.era_counts)
    
    def __getitem__(self, idx):
        era = self.era_counts.index[idx]
        data = self.df.loc[era]
        data = data.sample(self.era_counts.min()) # this is necessary to prevent memory leaks in the multiheadattention module
        return [
            torch.from_numpy(data.drop(columns='target').values), 
            torch.from_numpy(data[['target']].values)
        ]


class RandomData(Dataset):
    def __init__(self, df: pd.DataFrame):
        df = df.astype(float)
        self.x = torch.from_numpy(df.drop(columns=['target', 'era']).values).float().to(device)
        self.y = torch.from_numpy(df[['target']].values).float().to(device)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            self.x[idx], 
            self.y[idx],
        )


def download_datasets():
    napi = NumerAPI()
    napi.download_dataset(f"{DATA_VERSION}/train_int8.parquet")
    napi.download_dataset(f"{DATA_VERSION}/validation_int8.parquet")


@cached({})
def get_features():
    # retrieve the most important features based GBT model
    return pd.read_parquet('important_features.parquet').feature.tolist()


def get_train_df(features: list[str]):
    return pd.read_parquet(numerai_data_path / 'train_int8.parquet', columns=features + ['era', 'target'])


def get_validation_df(features):
    df = pd.read_parquet(numerai_data_path / 'validation_int8.parquet', columns=features + ['era', 'target'])
    return df[df['target'].notna()]


def validate(model: nn.Module, validation_df: pd.DataFrame):
    validation = torch.Tensor(validation_df[get_features()].values).to(device)
    validation_loader = DataLoader(validation, shuffle=False, batch_size=2048)
    model.eval()
    predictions = []
    for x in validation_loader:
        y = model(torch.Tensor([]), x)
        y = torch.tensor(y.detach(), device='cpu')
        predictions.append(y.numpy())

    validation_df['prediction'] = np.concatenate(predictions)
    era_corrs = validation_df.groupby('era').apply(
        lambda x: numerai_corr(x[['prediction']], x['target'])
    )
    era_corrs.to_parquet('era_corrs.parquet')
    return era_corrs


if __name__ == '__main__':
    main()