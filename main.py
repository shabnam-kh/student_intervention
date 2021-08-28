import pandas as pd
import sys

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from data_analyzer import explore_student_data, prepare_student_data, preprocess_features, reformat_labels
from classification import general_classification, tune_classifier_params, predict_by_best_estimator
from custom_logging import Logger


def main(argv):
    # debug mode
    # TODO
    argv = False
    if argv:
        Logger.turn_on()
    student_data = pd.read_csv("student-data.csv")

    # data exploration
    explore_student_data(student_data)

    # prepare data
    X_all, y_all = prepare_student_data(student_data)
    X_all = preprocess_features(X_all)
    y_all = reformat_labels(y_all)
    general_classification(X_all, y_all, [GaussianNB(), RandomForestClassifier(), SVC()],
                           [100, 200, 300])

    best_clf = tune_classifier_params(X_all, y_all, SVC(), parameters = [{'C':[1,10,50,100,200,300,400,500,1000,],
                         'gamma':[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
                         'kernel': ['rbf']}], train_size=100)

    predict_by_best_estimator(X_all, y_all, best_clf, train_size=100)


if __name__ == '__main__':
    main(sys.argv[0])
