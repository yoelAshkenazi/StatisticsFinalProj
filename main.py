import statistic_analysis as sa
from typing import List
import pandas as pd


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
    function_args = {'alpha': 0.05, "print_results": False, 'plot': True}

    df = sa.load_data()  # Load data

    # remember important columns
    important_cols = ["InterviewScore", "SkillScore", "PersonalityScore"]

    target_col = "HiringDecision"

    # check the distribution of the important columns
    # check_multiple_cols(df, important_cols, sa.analyze_distribution, **function_args)

    # check the dependence of the important columns on the target column
    check_multiple_cols(df, important_cols, sa.check_dependence, **function_args)
