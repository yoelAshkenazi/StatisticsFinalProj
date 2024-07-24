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

    target_col = "HiringDecision"

    # plot PCA
    # sa.plot_pca(df, 2)
    # sa.plot_pca(df, 3)

    # check the distribution of the important columns
    # check_multiple_cols(df, important_cols, sa.analyze_distribution, **function_args)

    # check the dependence of the important columns on the target column
    # check_multiple_cols(df, important_cols, sa.check_dependence, **function_args)

    # check the hypothesis with the xgboost model
    xgb_differences = []
    for i, feature in enumerate(regular_cols):
        print(f"Testing the theory of {feature} improvement with {function_args['model_name']}:")
        a_, b = sa.test_theory_feature_improvement(df, feature, **function_args)
        xgb_differences.append(list(np.array(b) - np.array(a_)))  # get the differences
        print('-'*40)

    # check the hypothesis with the neural network
    nn_diff = []
    function_args['model_name'] = 'nn'
    for i, feature in enumerate(regular_cols):
        print(f"Testing the theory of {feature} improvement with {function_args['model_name']}:")
        a_, b = sa.test_theory_feature_improvement(df, feature, **function_args)
        nn_diff.append(list(np.array(b) - np.array(a_)))  # get the differences
        print('-'*40)

    # perform the ANOVA test
    print("ANOVA test for XGB:")
    sa.test_theory_contribution(xgb_differences, **function_args)
    print('-' * 100)
    print("ANOVA test for NN:")
    sa.test_theory_contribution(nn_diff, **function_args)

    # check the biases.
    # for col in regular_cols:
    #     print(f"\nChecking the bias of {col} on the target column...")
    #     show_rates = True if col != "Age" else False
    #     sa.plot_feature_biases(df, col, show_rates)
    #     print('-'*40)

# todo: show some biases between the values of each categorical feature's values and the target column.
# todo: check the same results for other features (compare the accuracy of the model with and without the feature)
