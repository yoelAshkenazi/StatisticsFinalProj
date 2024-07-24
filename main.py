import numpy as np

import statistic_analysis as sa
from typing import List
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def check_multiple_cols(data: pd.DataFrame, columns: List[str], function: callable, **kwargs):
    """
    Check multiple columns of the data.
    :param data: the dataframe
    :param columns: the columns to check
    :param function: the function to apply
    :param kwargs: the keyword arguments to pass to the function
    :return:
    """
    for col in columns:
        a = function(data, col, **kwargs)
        if function.__name__ == 'analyze_distribution':
            print("\n")
        elif function.__name__ == 'check_dependence':
            print(f"\n{col} vs other columns:")
            print(a)
            # save the results
            a.to_csv(f'Analysis Plots/Dependency Results/{col}_dependence_with_others.csv')


def make_pca(data: pd.DataFrame, n_components: int):
    """
    Make PCA on the data.
    :param data: the data
    :param n_components: the number of components
    :return:
    """
    sa.plot_pca(data, n_components)


def check_feature_dependence(data: pd.DataFrame, features: List, **kwargs):
    """
    Check the dependence of the feature on the target column.
    :param data: the data
    :param features: list of features to check
    :param kwargs: the keyword arguments
    :return:
    """
    check_multiple_cols(data, features, sa.check_dependence, **kwargs)


def check_feature_distribution(data: pd.DataFrame, features: List[str], **kwargs):
    """
    Check the distribution of the features.
    :param data: the data
    :param features: the features to check
    :param kwargs: the keyword arguments
    :return:
    """
    check_multiple_cols(data, features, sa.analyze_distribution, **kwargs)


def check_features_improvement(features: List[str], **kwargs) -> (List, List):
    """
    Check the improvement of the feature on the target column.
    :param features: list of features to check
    :param kwargs: the keyword arguments
    :return:
    """
    # check the hypothesis with the xgboost model
    xgb_diff = []
    normal_xgb_diffs = {}  # store the names of normal differences
    for i, feature in enumerate(features):
        print(f"Testing the theory of '{feature}' improvement with {kwargs['model_name']}:")
        a_, b_, is_normal = sa.test_theory_feature_improvement(df, feature, **kwargs)
        xgb_diff.append(list(np.array(b_) - np.array(a_)))  # get the differences
        normal_xgb_diffs[feature] = is_normal
        print('-' * 40)

    # check the hypothesis with the neural network
    nn_diff = []
    normal_nn_diffs = {}  # store the names of normal differences
    function_args['model_name'] = 'nn'
    for i, feature in enumerate(features):
        print(f"Testing the theory of '{feature}' improvement with {kwargs['model_name']}:")
        a_, b_, is_normal = sa.test_theory_feature_improvement(df, feature, **kwargs)
        nn_diff.append(list(np.array(b_) - np.array(a_)))  # get the differences
        normal_nn_diffs[feature] = is_normal
        print('-' * 40)

    return (xgb_diff, normal_xgb_diffs), (nn_diff, normal_nn_diffs)


def test_anova(results: List, normal_flags: List, **kwargs):
    """
    Test the results' improvements using the ANOVA test.
    :param results: the results to test
    :param normal_flags: the normal flags
    :param kwargs: the keyword arguments
    :return:
    """
    print("Analyzing the results using ANOVA/Kruskal-Wallis test...")
    sa.test_theory_contribution(results, normal_flags, **kwargs)


def check_biases(data: pd.DataFrame, features: List[str]):
    """
    Check the biases in the data.
    :param data: the data
    :param features: the features to check
    :return:
    """
    for feature in features:
        print(f"Checking the bias of {feature} on the target column...")
        sa.plot_feature_biases(data, feature, True if feature not in ['Age', 'DistanceFromCompany'] else False)
        print('-' * 40)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # set up parameters
    ALPHA = 0.05
    function_args = {'alpha': 0.05, "print_results": True, 'plot': True, 'model_name': 'xgb'}

    df = sa.load_data()  # Load data

    # remember important columns
    important_cols = ["InterviewScore", "SkillScore", "PersonalityScore"]
    # regular columns
    regular_cols = ["Age", "Gender", "EducationLevel", "ExperienceYears", "PreviousCompanies"]

    misc_cols = ['DistanceFromCompany']

    target_col = "HiringDecision"

    ALL_COLS = important_cols + regular_cols + misc_cols

    """
    check the distribution of the features
    """
    # check_feature_distribution(df, ALL_COLS, **function_args)

    """
    Test the dependence of the features on the target column
    """
    # check_feature_dependence(df, ALL_COLS, **function_args)

    """
    Check biases and tendencies in the data
    """
    # check_biases(df, ALL_COLS)

    """
    Check the improvement of the features on the target column
    """
    # xgb_results, nn_results = check_features_improvement(ALL_COLS, **function_args)

    """
    Perform the ANOVA test on the results of the features
    """
    # test_anova(xgb_results[0], xgb_results[1].values(), **function_args)
    # test_anova(nn_results[0], nn_results[1].values(), **function_args)

# todo: show some biases between the values of each categorical feature's values and the target column.
# todo: check the same results for other features (compare the accuracy of the model with and without the feature)
