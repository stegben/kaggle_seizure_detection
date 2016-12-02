import sys
from pprint import pprint

import numpy as np
import pandas as pd
import joblib

import xgboost as xgb
# from sklearn.cross_validation import PredefinedSplit
# from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score

from tqdm import tqdm

# TUNED_PARAMS = [
#                 {'n_estimators': [500],
#                  'criterion': ['gini', 'entropy'],
#                  'max_depth': [None],
#                  'max_features': [0.01]}
#                ]

search_models = [
    {
        'min_child_weight': 1,
        'learning_rate': 0.03,
        'colsample_bytree': 0.5,
        'subsample': 0.7,
        'gamma': 1,
        'silent': 0,
        'seed': 1234,
        # 'booster': 'gblinear',
        # 'booster': 'gbtree',
        'max_depth': 6,
        'objective': "binary:logistic", # 'rank:pairwise'
        'nthread': 10,
    },
    {
        'min_child_weight': 1,
        'learning_rate': 0.03,
        'colsample_bytree': 0.6,
        'subsample': 0.7,
        'gamma': 1,
        'silent': 0,
        'seed': 1234,
        # 'booster': 'gblinear',
        # 'booster': 'gbtree',
        'max_depth': 6,
        'objective': "binary:logistic", # 'rank:pairwise'
        'nthread': 10,
    },
    {
        'min_child_weight': 1,
        'learning_rate': 0.03,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'gamma': 1,
        'silent': 0,
        'seed': 1234,
        # 'booster': 'gblinear',
        # 'booster': 'gbtree',
        'max_depth': 6,
        'objective': "binary:logistic", # 'rank:pairwise'
        'nthread': 10,
    },
    {
        'min_child_weight': 1,
        'learning_rate': 0.03,
        'colsample_bytree': 0.8,
        'subsample': 0.7,
        'gamma': 1,
        'silent': 0,
        'seed': 1234,
        # 'booster': 'gblinear',
        # 'booster': 'gbtree',
        'max_depth': 6,
        'objective': "binary:logistic", # 'rank:pairwise'
        'nthread': 10,
    },
]

def combine_fname_pred(pred, fnames):
    return pd.DataFrame({'fnames': fnames, 'pred': pred})


def main():
    data_fname = sys.argv[1]
    sub_fname = sys.argv[2]

    split_data = joblib.load(data_fname)

    df_subtrain = split_data['subtrain']
    df_validation = split_data['validation']
    df_test = split_data['test']
    # import ipdb; ipdb.set_trace()

    df_subtrain = df_subtrain.fillna(df_subtrain.drop('fnames', axis=1).mean().to_dict())
    df_validation = df_validation.fillna(df_validation.drop('fnames', axis=1).mean().to_dict())
    df_test = df_test.fillna(df_test.drop('fnames', axis=1).mean().to_dict())
    # import ipdb; ipdb.set_trace()

    x_subtrain = df_subtrain.drop(['label', 'fnames'], axis=1).values
    x_validation = df_validation.drop(['label', 'fnames'], axis=1).values

    y_subtrain = df_subtrain['label'].values
    y_validation = df_validation['label'].values
    y_validation_agg = df_validation.groupby('fnames')['label'].mean().values

    x_test = df_test.drop(['fnames'], axis=1).values

    # f_subtrainold = [-1 for i in range(len(y_subtrain))]
    # validation_fold = [0 for i in range(len(y_validation))]
    # test_fold = f_subtrainold + validation_fold
    # ps = PredefinedSplit(test_fold)

    score = []
    best_mean_model = None
    best_mean_model_param = None
    best_mean_auc = 0.
    best_ms_model = None
    best_ms_model_param = None
    best_ms_auc = 0.

    for model_param in tqdm(search_models):
        pprint(model_param)

        clf = xgb.XGBClassifier(n_estimators=2000, **model_param)
        clf.fit(
            x_subtrain, y_subtrain,
            eval_set=[(x_subtrain, y_subtrain), (x_validation, y_validation)],
            eval_metric='auc',
            early_stopping_rounds=30,
            verbose=True,
        )

        # clf = model_param['model'](**model_param['params'])
        # clf.fit(x_subtrain, y_subtrain)

        pred_validation = clf.predict_proba(x_validation)
        pred_with_fname = combine_fname_pred(pred_validation[:, 1], df_validation['fnames'])
        mean_pred = pred_with_fname.groupby('fnames')['pred'].mean().values
        ms_pred = pred_with_fname.groupby('fnames')['pred'].agg(lambda x: np.mean(np.square(x))).values

        raw_score = roc_auc_score(y_validation, pred_validation[:, 1])
        mean_score = roc_auc_score(y_validation_agg, mean_pred)
        ms_score = roc_auc_score(y_validation_agg, ms_pred)
        print('raw AUC score: {}'.format(raw_score))
        print('mean AUC score: {}'.format(mean_score))
        print('mean square AUC score: {}'.format(ms_score))

        if mean_score > best_mean_auc:
            best_mean_auc = mean_score
            best_mean_model_param = model_param
        if ms_score > best_ms_auc:
            best_ms_auc = ms_score
            best_ms_model_param = model_param
    print(best_mean_model_param)
    print(best_ms_model_param)
    clf = xgb.XGBClassifier(n_estimators=300, **best_mean_model_param)
    clf.fit(np.concatenate((x_subtrain, x_validation), axis=0), np.concatenate((y_subtrain, y_validation), axis=0))
    pred_mean = clf.predict_proba(x_test)
    df_test['Class'] = pred_mean[:, 1]
    df_test['fnames'] = 'new_' + df_test['fnames']
    df_test.groupby('fnames')['Class'].mean().to_csv('mean_' + sub_fname, index_label='File', header=True)

    pred_ms = clf.predict_proba(x_test)
    df_test['Class'] = pred_ms[:, 1]
    df_test.groupby('fnames')['Class'].agg(lambda x: np.mean(np.square(x))).to_csv('mean_square_' + sub_fname, index_label='File', header=True)

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
