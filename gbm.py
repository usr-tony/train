from train import get_train_df, get_features, BUCKET_NAME
import lightgbm
import os


def main():
    max_depth = 6
    model = lightgbm.LGBMRegressor(
        n_estimators = 30000,
        num_leaves = 2 ** max_depth - 1,
        max_depth = max_depth,
        colsample_bytree = 0.15,
        learning_rate = 0.001,
    )
    train_df = get_train_df()
    features = get_features()
    print('starting train')
    model.fit(
        train_df[list(features)],
        train_df.target
    )
    model_file = f'model_gbm_large.txt'
    model.booster_.save_model(model_file)
    os.system(f'gcloud storage cp {model_file} gs://{BUCKET_NAME}')


if __name__ == '__main__':
    main()