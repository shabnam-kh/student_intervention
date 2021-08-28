import pandas as pd
import sys

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from student_analyzer import StudentAnalyzer
from classification import general_classification, tune_classifier_params, predict_by_best_estimator
from custom_logging import Logger


def main(argv):
    # debug mode
    # TODO
    argv = False
    if argv:
        Logger.turn_on()
    student_data = pd.read_csv("student-data.csv")
    student_analyzer = StudentAnalyzer(student_data)

    # data exploration
    student_analyzer.explore_student_data()

    # prepare data
    student_analyzer.prepare_student_data()
    student_analyzer.preprocess_features()
    student_analyzer.reformat_labels()
    general_classification(student_analyzer, [GaussianNB(), RandomForestClassifier(), SVC()],
                           [100, 200, 300])

    best_clf = tune_classifier_params(student_analyzer, SVC(), parameters = [{'C':[1,10,50,100,200,300,400,500,1000,],
                         'gamma':[1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
                         'kernel': ['rbf']}], train_size=100)

    predict_by_best_estimator(student_analyzer, best_clf, train_size=100)


if __name__ == '__main__':
    main(sys.argv[0])
