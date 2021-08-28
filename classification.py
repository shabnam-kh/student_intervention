from time import time
import pandas as pd
from sklearn.metrics import f1_score

from custom_logging import Logger

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
    _logger.info( "prediction done in {:.4f} seconds".format(period))
    return period, f1_score(target, y_predict, pos_label='yes')


def train_test_predict(clf, X_train, y_train, X_test, y_test):
    _logger.info( "Training {} using train size {}".format(clf.__class__.__name__, len(X_train)))

    train_time = train_classifier(clf, X_train, y_train)
    _, train_score = predict_labels(clf, X_train, y_train)
    test_time, test_score = predict_labels(clf, X_test, y_test)
    _logger.info( "f1 score for train data is {:.4f}".format(train_score))
    _logger.info( "f1 score for test data is {:.4f}".format(test_score))
    return train_time, train_score, test_time, test_score


def general_classification(student_analyzer, clf_list, train_size_list):
    for clf in clf_list:
        for size in train_size_list:
            df = pd.DataFrame(columns=['Training_Size',
                                       'Testing_Size',
                                       'Training_Time',
                                       'Prediction_Time',
                                       'F1_Training_Score',
                                       'F1_Testing_Score'])

            X_train, X_test, y_train, y_test = student_analyzer.shuffle_split_data(train_size=size)

            cycles = 10
            for i in range(cycles):
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



