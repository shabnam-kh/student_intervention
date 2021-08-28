import pandas as pd
import sys

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from student_analyzer import StudentAnalyzer
from classification import general_classification
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
    general_classification(student_analyzer, [GaussianNB(), RandomForestClassifier(), SVC()],
                           [100, 200, 300])


if __name__ == '__main__':
    main(sys.argv[0])
