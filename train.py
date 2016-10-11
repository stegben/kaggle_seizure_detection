import sys
from pprint import pprint

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import PredefinedSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score

TUNED_PARAMS = [
                {'n_estimators': [1000],
                 'criterion': ['gini'],
                 'max_depth': [5],
                 'max_features': ['auto']}
               ]

def get_test_split():
    return test_split

def main():
    data_fname = sys.argv[1]
    sub_fname = sys.argv[2]

    split_data = joblib.load(data_fname)

    df_subtrain = split_data['subtrain']
    df_validation = split_data['validation']
    df_test = split_data['test']

    x_subtrain = df_subtrain.drop(['label', 'fnames'], axis=1).values
    x_validation = df_validation.drop(['label', 'fnames'], axis=1).values

    y_subtrain = df_subtrain['label'].values
    y_validation = df_validation['label'].values

    subtrain_fold = [-1 for i in range(len(y_subtrain))]
    validation_fold = [0 for i in range(len(y_validation))]
    test_fold = subtrain_fold + validation_fold
    ps = PredefinedSplit(test_fold)

    x = np.concatenate((x_subtrain, x_validation), axis=0)
    y = np.concatenate((y_subtrain, y_validation), axis=0)

    clf_search = GridSearchCV(RandomForestClassifier(n_jobs=20, min_samples_leaf=2, min_samples_split=4),
                   param_grid=TUNED_PARAMS,
                   scoring='roc_auc',
                   n_jobs=1,
                   verbose=5,
                   cv=ps,
                   refit=True
                  )
    clf_search.fit(x, y)
    pprint(clf_search.grid_scores_)
    clf = clf_search.best_estimator_
    print(clf.feature_importances_)

    x_test = df_test.drop(['fnames'], axis=1).values
    pred = clf.predict_proba(x_test)
    df_test['Class'] = pred[:, 1]
    df_test.groupby('fnames')['Class'].mean().to_csv('mean_' + sub_fname, index_label='File', header=True)
    df_test.groupby('fnames')['Class'].max().to_csv('max_' + sub_fname, index_label='File', header=True)
    df_test.groupby('fnames')['Class'].agg(lambda x: np.mean(np.square(x)).to_csv('mean_square_' + sub_fname, index_label='File', header=True)


if __name__ == '__main__':
    main()
