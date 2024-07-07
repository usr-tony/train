import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import pandas as pd
from pathlib import Path
from functools import cache
from numerapi import NumerAPI
import numpy as np
import json
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel


sdpa_kernel(SDPBackend.FLASH_ATTENTION).__enter__()

EMBED_DIM = 32
FEATURE_SET = 'small'

BUCKET_NAME = 'train1230'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device=}')
DATA_VERSION = 'v4.3'
numerai_data_path = Path(DATA_VERSION)
napi = NumerAPI()


def main():
    model = Model(
        nfeatures=len(get_features())
    ).to(device)
    try:
        model.load_state_dict(torch.load('model.pkl'))
        print('loaded pretrained model')
    except Exception as e:
        print(e)
        pass

    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-4,
    )
    train_df = get_train_df(get_features())
    train = DataByEra(train_df)
    train_loader = DataLoader(train, batch_size=1)
    best_corr = -np.inf
    for epoch in range(20):
        print(f'epoch {epoch}')
        model.train()
        for [x], [labels] in tqdm(train_loader):
            optimizer.zero_grad()
            y = model(x)
            loss = loss_func(y, labels) * 0.1 - corr(y, labels) * 0.9
            loss.backward()
            optimizer.step()

        if epoch < 5:
            print('skipping validation for first 5 epochs')
            continue

        with torch.no_grad():
            era_corr_sum = evaluate(model, epoch).sum()

        print('sum of correlations \n', era_corr_sum)
        if best_corr < era_corr_sum:
            best_corr = era_corr_sum
            torch.save(model.state_dict(), 'model.pkl')
        else:
            print('no improvement')
            return


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
    def __init__(self, dim=EMBED_DIM, ff_inner_dim=EMBED_DIM * 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_inner_dim),
            nn.GELU(), # GEGLU is used in the paper
            nn.Dropout(0.2),
            nn.Linear(ff_inner_dim, dim)
        )

    def forward(self, x):
        attn_in = self.norm(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = attn_out + x # should the residual be applied to before or after the norm?
        return self.ff(x) + x # residual


class Model(nn.Module):
    def __init__(self, nfeatures, embed_dim=EMBED_DIM):
        super().__init__()
        self.embedding = Embedding(nfeatures)
        self.transformer = Transformer(embed_dim)
        self.inter_sample_transformer = Transformer(embed_dim * nfeatures, ff_inner_dim=embed_dim * 200)
        self.final = nn.Sequential(
            # layernorm here?
            nn.Linear(embed_dim * nfeatures, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = rearrange(x, 'b n d -> 1 b (n d)')
        x = self.inter_sample_transformer(x)

        return self.final(x).squeeze(0)


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
            torch.from_numpy(data.drop(columns='target').values).to(device),
            torch.from_numpy(data[['target']].values).to(device)
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


@cache
def get_features() -> tuple[str]:
    path = f"{DATA_VERSION}/features.json"
    napi.download_dataset(path)
    with open(path) as f:
        features = json.loads(f.read())

    return tuple(features['feature_sets'][FEATURE_SET])


def get_train_df(features: tuple[str]=None):
    features = features or get_features()
    path = f"{DATA_VERSION}/train_int8.parquet"
    napi.download_dataset(path)
    return pd.read_parquet(path, columns=['era', 'target'] + list(features))


@cache
def get_validation_df(features: tuple[str]=None):
    features = features or get_features()
    path = f"{DATA_VERSION}/validation_int8.parquet"
    napi.download_dataset(path)
    df = pd.read_parquet(path, columns=['era', 'target'] + list(features))
    return df[df['target'].notna()]


def evaluate(model: nn.Module, epoch: int, validation_df: pd.DataFrame=None):
    model.eval()
    if validation_df is None:
        validation_df = get_validation_df()

    validation_df = validation_df.copy(deep=False)
    predictions = []
    for era, group in tqdm(validation_df.groupby('era')):
        x = group.loc[:, get_features()].values.astype(np.int8)
        x = torch.from_numpy(x).to(device)
        group['prediction'] = model(x).detach().cpu().numpy()
        predictions.append(group['prediction'])

    validation_df['prediction'] = pd.concat(predictions)
    validation_df[['prediction']].to_parquet(f'predictions_epoch_{epoch}.parquet')
    # calculate corr per era
    return (
        validation_df[['prediction', 'target', 'era']]
        .groupby('era')
        .apply(lambda x: x['prediction'].corr(x['target']))
    )


def corr(a: torch.Tensor, b: torch.Tensor):
    da = a - a.mean()
    db = b - b.mean()
    numer = torch.sum(da * db)
    denom = torch.sqrt(torch.sum(da ** 2) * torch.sum(db ** 2))
    return numer / denom


if __name__ == '__main__':
    main()