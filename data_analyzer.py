import pandas as pd
from sklearn.model_selection import train_test_split

from custom_logging import Logger

_logger = Logger('DataAnalyzer')


def explore_student_data(student_data):
    student_counts = student_data.shape[0]
    feature_counts = student_data.shape[1]
    student_passed_counts = student_data[student_data['passed'] == 'yes'].shape[0]
    student_failed_counts = student_data[student_data['passed'] == 'no'].shape[0]
    passed_percentage = student_passed_counts * 100 / student_counts  # to figureout if data is unbalanced

    _logger.info("student count is {}".format(student_counts))
    _logger.info("feature count is {}".format(feature_counts))
    _logger.info("passed student count is {}".format(student_passed_counts))
    _logger.info("failed student count is {}".format(student_failed_counts))
    _logger.info("passed_percentage is {}%".format(passed_percentage))


def prepare_student_data(student_data):
    feature_columns = list(student_data.columns[:-1])
    label_column = student_data.columns[-1]

    _logger.info("list of features {}".format(feature_columns))
    _logger.info("label column is {}".format(label_column))

    X_all = student_data[feature_columns]
    y_all = student_data[label_column]

    _logger.info("features sample value {}".format(X_all.head()))
    return X_all, y_all


def preprocess_features(X_all):
    """
     Preprocesses the student data and converts non-numeric binary variables into
     binary (0/1) variables. Converts categorical variables into dummy variables.
    """
    # Initialize new output DataFrame
    processed_features = pd.DataFrame(index=X_all.index)

    # Investigate each feature column for the data
    for feature_name, feature_value in X_all.iteritems():

        # Investigate each feature column for the data
        if feature_value.dtype == object:
            feature_value = feature_value.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if feature_value.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            feature_value = pd.get_dummies(feature_value, prefix=feature_name)
        processed_features = processed_features.join(feature_value)

    _logger.info("processd features, count is {}, features are {}".format(len(processed_features.columns),
                                                                          list(processed_features.columns)))
    return processed_features


def reformat_labels(y_all):
    return y_all.replace(['yes', 'no'], [1, 0])


def shuffle_split_data(X_all, y_all, train_size):
    test_size = X_all.shape[0] - train_size
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=0)
    _logger.info("splited data, X_train has {} samples".format(X_train.shape[0]))
    _logger.info("splited data, y_train has {} samples".format(y_train.shape[0]))
    return X_train, X_test, y_train, y_test
