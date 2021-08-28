from time import time

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from custom_logging import Logger
from data_analyzer import shuffle_split_data

_logger = Logger('Classification')


def train_classifier(clf, X_train, y_train):
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    period = end - start
    _logger.info("train model in {:.4f} seconds".format(period))
    return period


def predict_labels(clf, features, target):
    start = time()
    y_predict = clf.predict(features)
    end = time()
    period = end - start
    _logger.info("prediction done in {:.4f} seconds".format(period))
    return period, f1_score(target, y_predict, pos_label=1)


def train_test_predict(clf, X_train, y_train, X_test, y_test):
    _logger.info("Training {} using train size {}".format(clf.__class__.__name__, len(X_train)))

    train_time = train_classifier(clf, X_train, y_train)
    _, train_score = predict_labels(clf, X_train, y_train)
    test_time, test_score = predict_labels(clf, X_test, y_test)
    _logger.info("f1 score for train data is {:.4f}".format(train_score))
    _logger.info("f1 score for test data is {:.4f}".format(test_score))
    return train_time, train_score, test_time, test_score


def general_classification(X_all, y_all, clf_list, train_size_list):
    for clf in clf_list:
        for size in train_size_list:
            df = pd.DataFrame(columns=['Training_Size',
                                       'Testing_Size',
                                       'Training_Time',
                                       'Prediction_Time',
                                       'F1_Training_Score',
                                       'F1_Testing_Score'])

            X_train, X_test, y_train, y_test = shuffle_split_data(X_all, y_all, train_size=size)

            cycles = 10
            for i in range(0, cycles):
                train_time, f1_score_train, pred_time_test, f1_score_test = \
                    train_test_predict(clf, X_train, y_train, X_test, y_test)
                df = df.append({'Training_Size': X_train.shape[0],
                                'Testing_Size': X_test.shape[0],
                                'Training_Time': train_time,
                                'Prediction_Time': pred_time_test,
                                'F1_Training_Score': f1_score_train,
                                'F1_Testing_Score': f1_score_test},
                               ignore_index=True)
            df_size = df[(df.Training_Size == size)]
            df_size_mean = df_size.mean()
            print "**********************************************************"
            print "Mean Statistics: {}".format(clf.__class__.__name__)
            print df_size_mean
            print "**********************************************************"


def tune_classifier_params(X_all, y_all, clf, parameters, train_size):
    f1_scores = []
    gamma = []
    c = []
    cycles = 10
    for i in range(0, cycles):
        grid_clf = GridSearchCV(clf, parameters, scoring='f1')
        X_train, X_test, y_train, y_test = shuffle_split_data(X_all, y_all, train_size=train_size)

        grid_clf.fit(X_train, y_train)
        f1_scores.append(grid_clf.score(X_test, y_test))
        gamma.append(grid_clf.best_params_['gamma'])
        c.append(grid_clf.best_params_['C'])
        best_clf = grid_clf.best_estimator_

    df_f1 = pd.Series(f1_scores)
    df_gamma = pd.Series(gamma)
    df_c = pd.Series(c)

    _logger.info("f1_score {}".format(df_f1))
    print "mean f1 score {}".format(df_f1.mean())
    _logger.info( "gamma {}".format(df_gamma[0]))
    print "gamma {}".format(df_gamma[0])
    _logger.info( "c param {}".format(df_c[0]))
    print "c param {}".format(df_c[0])

    return best_clf


def predict_by_best_estimator(X_all, y_all, clf, train_size):
    X_train, X_test, y_train, y_test = shuffle_split_data(X_all, y_all, train_size=train_size)
    _, f1_train = predict_labels(clf, X_train, y_train)
    _, f1_test = predict_labels(clf, X_test, y_test)
    print "Tuned model has a training F1 score of {:.4f}.".format(f1_train)
    print "Tuned model has a testing F1 score of {:.4f}.".format(f1_test)


