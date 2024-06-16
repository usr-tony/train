import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from tab_transformer_pytorch import FTTransformer
from tqdm import tqdm
from train import *

pd.options.display.float_format = lambda x: f'{x:.5f}'
DATA_VERSION = 'v4.3'
numerai_data_path = Path(DATA_VERSION)
EMBED_DIM = 64
BATCH_SIZE = 2 ** 11
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device=}')


def main():
    model = FTTransformer(
        categories=[],
        num_continuous=len(get_features()),
        dim=EMBED_DIM,
        depth=2,
        heads=8,
        attn_dropout=0.2,
        ff_dropout=0.1,
    )
    loss_func = nn.MSELoss() 
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=1e-4,
        nesterov=True,
        momentum=0.3,
    )
    train_df = get_train_df(get_features())
    train = RandomData(train_df)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE)
    model.to(device)
    for epoch in range(10):
        model.train()
        for i, (x, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            y = model(torch.Tensor([]), x)
            loss = loss_func(y, labels) - corr(y, labels)
            loss.backward()
            optimizer.step()

        torch.save(model, f'model_epoch_{epoch}.pkl')

main()