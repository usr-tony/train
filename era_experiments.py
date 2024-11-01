import pandas as pd
from functools import cache
from numerai_tools.scoring import numerai_corr
import json
from numerapi import NumerAPI
import lightgbm
from joblib import delayed, Parallel
from train import get_train_df, get_validation_df, get_features
import numpy as np


napi = NumerAPI()

FEATURE_SET = 'medium'
DATA_VERSION = 'v4.3'
predictions_file = '/kaggle/input/prediction-sets/medium_3x_layers.parquet'
GBM_MAX_DEPTH = 6


def main():
    Parallel(n_jobs=1)(
        delayed(train_model_era)(era, group, list(get_features())) for era, group in get_train_df().groupby('era')
    )
    print('starting getting train predictions')
    validation_df = get_validation_df()
    train_era = train_df['era'].unique()
    res = Parallel(n_jobs=15)(
        get_preds(era, validation_df) for era in train_df['era'].unique()
    )
    res_df = pd.concat(res, axis='columns')
    res_df.to_parquet('validation_era_predictions.parquet')


def metrics_from_file(file_name):
    df = pd.read_parquet(file_name)
    era_corr = get_era_corr(df)
    return metrics(era_corr)


def get_era_corr(preds: pd.Series, df=None):
    if df is None:
        df = get_validation_df()
    df.dropna(subset='target', inplace=True)
    df['model_prediction'] = preds
    return df.groupby('era').apply(lambda x: x.model_prediction.corr(x.target))


def metrics(era_corr):
    print(era_corr.describe())
    print('total adj:', era_corr.mean() * 522)
    print('sharpe:', era_corr.mean() / era_corr.std())
    print('only negative eras:')
    print(era_corr[era_corr < 0].describe())
    

def neutralize(
    df: pd.DataFrame,
    neutralizers: pd.DataFrame,
    proportion: float = 1.0, 
):
    '''
    Arguments:
        df: pd.DataFrame - the data with columns to neutralize
        neutralizers: pd.DataFrame - the neutralizer data with features as columns
        proportion: float - the degree to which neutralization occurs
    '''
    args = (df, neutralizers, proportion)
    try:
        return neutralize_v2(*args)
    except np.linalg.LinAlgError:
        return neutralize_og(*args)


def neutralize_og(
    df: pd.DataFrame,
    neutralizers: pd.DataFrame,
    proportion: float = 1.0,
) -> pd.DataFrame:
    """Neutralize each column of a given DataFrame by each feature in a given
    neutralizers DataFrame. Neutralization uses least-squares regression to
    find the orthogonal projection of each column onto the neutralizers, then
    subtracts the result from the original predictions.

    Args:
        df: pd.DataFrame - the data with columns to neutralize
        neutralizers: pd.DataFrame - the neutralizer data with features as columns
        proportion: float - the degree to which neutralization occurs

    Returns:
        pd.DataFrame - the neutralized data
    """
    assert not neutralizers.isna().any().any(), "Neutralizers contain NaNs"
    assert len(df.index) == len(neutralizers.index), "Indices don't match"
    assert (df.index == neutralizers.index).all(), "Indices don't match"
    df[df.columns[df.std() == 0]] = np.nan
    df_arr = df.values
    neutralizer_arr = neutralizers.values
    neutralizer_arr = np.hstack(
        # add a column of 1s to the neutralizer array in case neutralizer_arr is a single column
        (neutralizer_arr, np.ones(len(neutralizer_arr)).reshape(-1, 1))
    )
    inverse_neutralizers = np.linalg.pinv(neutralizer_arr, rcond=1e-6)
    adjustments = proportion * neutralizer_arr.dot(inverse_neutralizers.dot(df_arr))
    neutral = df_arr - adjustments
    return pd.DataFrame(neutral, index=df.index, columns=df.columns)


def neutralize_v2(df, features, prop=1):
    df[df.columns[df.std() == 0]] = np.nan # columns where std is 0 has all values set to np.nan
    features['_feature_ones'] = 1
    if len(features) >= len(features.columns):
        inverse = np.linalg.inv(features.T @ features) @ features.T
    else:
        inverse = features.T @ np.linalg.inv(features @ features.T)

    adjustments = features.values @ (inverse @ df.values)
    neutral = df.values - prop * adjustments.values
    return pd.DataFrame(neutral, index=df.index, columns=df.columns)


def train_model_era(era, group):
    model = lightgbm.LGBMRegressor(
        n_estimators = 15,
        num_leaves = 2 ** GBM_MAX_DEPTH - 1,
        max_depth = GBM_MAX_DEPTH,
        colsample_bytree = 0.4,
        learning_rate = 0.01,
        n_jobs=2,
    )
    model.fit(group[features], group['target'])
    model.booster_.save_model(f'model_era_{era}.txt')




@delayed
def get_preds(era: str, df):
    model = lightgbm.Booster(model_file=f'model_era_{era}.txt')
    model_preds = []
    for indices in np.array_split(df.index, 20):
        xgroup = df.loc[indices, features]
        preds = model.predict(xgroup)
        model_preds.append(pd.Series(preds, index=indices))
        
    print(era, 'done')
    return pd.concat(model_preds).rename(era).astype(np.float32)


if __name__ == '__main__':
    main()