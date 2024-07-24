import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import List

# Constants
BLUE, RED = '#5F4B8B', '#E69A8D'


def load_data():
    """
    Load data from csv file
    :return:
    """
    data = pd.read_csv('data.csv')  # Load data from csv file to pandas dataframe.
    return data


def plot_pca(data: pd.DataFrame, n_components: int = 2):
    """
    Plots the PCA results of the data (either 2d or 3d).
    :param data: the data to plot
    :param n_components: the number of components to plot (either 2 or 3)
    :return:
    """

    assert n_components in [2, 3], "n_components must be either 2 or 3"

    # Perform PCA
    pca = PCA(n_components=n_components)

    # Fit and transform the data.
    principal_components = pca.fit_transform(data.iloc[:, : -1].copy())

    # calculate the explained variance ratio
    explained_variance_ratio = np.sum(pca.explained_variance_ratio_)

    # Create a new dataframe with the principal components.
    principal_df = pd.DataFrame(data=principal_components)

    # initialize the colors.
    colors = {0: BLUE, 1: RED}
    labels = {0: 'Reject', 1: 'Accept'}

    if n_components == 2:
        # Plot the PCA results in 2D
        plt.figure()  # Create a new figure

        # Scatter plot the principal components
        plt.scatter(principal_df.iloc[:, 0], principal_df.iloc[:, 1], c=data.iloc[:, -1].map(colors))

        plt.grid()  # Add grid
        plt.xlabel('Principal Component 1')  # Add x label
        plt.ylabel('Principal Component 2')  # Add y label
        plt.title(f'PCA 2D \nExplained Variance: {(explained_variance_ratio * 100):.3f}%')  # Add title

        # Create legend handles
        handles = [plt.Line2D(['Reject'], ['Accept'], marker='o', color='w', markerfacecolor=col,
                              markersize=10, label=labels[label]) for label, col in colors.items()]
        plt.legend(handles=handles, title='Label')
        # save the plot
        plt.savefig('Analysis Plots/pca_2d.png')

        plt.show()  # Show the plot

    else:
        # Plot the PCA results in 3D
        fig = plt.figure()  # Create a new figure
        ax = fig.add_subplot(111, projection='3d')  # Add 3D subplot

        # Scatter plot the principal components
        ax.scatter(principal_df.iloc[:, 0], principal_df.iloc[:, 1], principal_df.iloc[:, 2],
                   c=data.iloc[:, -1].map(colors))
        ax.set_xlabel('Principal Component 1')  # Add x label
        ax.set_ylabel('Principal Component 2')  # Add y label
        ax.set_zlabel('Principal Component 3')  # Add z label
        plt.title(f'PCA 3D \nExplained Variance: {(explained_variance_ratio * 100):.3f}%')  # Add title

        # Create legend handles
        handles = [plt.Line2D(['Reject'], ['Accept'], marker='o', color='w', markerfacecolor=col,
                              markersize=10, label=labels[label]) for label, col in colors.items()]
        plt.legend(handles=handles, title='Label')  # Add legend

        # save the plot
        plt.savefig('Analysis Plots/pca_3d.png')

        plt.show()  # Show the plot


def plot_correlation_matrix(data: pd.DataFrame):
    """
    Plots the correlation matrix of data.
    :param data: the data to plot
    :return:
    """
    # Create a correlation matrix
    corr = data.corr()

    # round the correlation matrix to 2 decimal places
    corr = corr.round(2)

    # Create a heatmap of the correlation matrix
    plt.figure()  # Create a new figure
    sns.heatmap(corr, annot=True, cmap='flare', cbar=True)  # Create a heatmap
    plt.title('Correlation Matrix')  # Add title

    # save the plot
    plt.savefig('Analysis Plots/correlation_matrix.png')

    plt.show()  # Show the plot


def analyze_distribution(data_: pd.DataFrame, feature_: str, **kwargs):
    """
    Plots the distribution of features in the data.
    :param data_: the data to plot
    :param feature_: the feature to plot
    :param kwargs: the keyword arguments
    :return:
    """

    # Constants
    ALPHA = kwargs.get('alpha', 0.05)  # Get the alpha value
    plot = kwargs.get('plot', True)  # Get the plot value

    feature = data_[feature_]  # Get the feature to plot

    # Perform the Kolmogorov-Smirnov test to compare the feature distribution to
    # a normal distribution and a uniform distribution. Note that we have enough
    # data points to perform the 1-sample Kolmogorov-Smirnov test.
    ks_stat, ks_p = stats.kstest(feature, 'norm', args=(np.mean(feature), np.std(feature)))
    ks_stat_uniform, ks_p_uniform = stats.kstest(feature, 'uniform',
                                                 args=(np.min(feature), np.max(feature)))

    # Show the distribution of the feature
    if plot:
        # Create a figure with 2 subplots, each comparing the
        # feature distribution to a different distribution using kde plots
        fig, axs = plt.subplots(2, 1, figsize=(15, 15))

        # Plot the feature distribution vs normal distribution
        sns.kdeplot(feature, color=BLUE, ax=axs[0], label=feature_)

        # Plot the normal distribution
        std = np.std(feature)
        steps = np.linspace(np.min(feature) - std, np.max(feature) + std, len(feature))
        axs[0].plot(steps, stats.norm.pdf(steps, np.mean(feature), std), color=RED,
                    label='Normal Distribution')

        # Add title and legend
        axs[0].set_title(f'{feature_} vs Normal Distribution \nPV: {ks_p:.5f}')
        axs[0].legend()  # Add legend
        axs[0].set_xlabel(feature_)  # Add x label
        axs[0].set_ylabel('Density')  # Add y label

        # Plot the feature distribution vs uniform distribution
        sns.kdeplot(feature, color=BLUE, ax=axs[1], label=feature_)

        # Plot the uniform distribution
        axs[1].plot(steps, stats.uniform.pdf(steps, np.min(feature), np.max(feature)), color=RED,
                    label='Uniform Distribution')

        # Add title and legend
        axs[1].set_title(f'{feature_} vs Uniform Distribution \nPV: {ks_p_uniform:.5f}')
        axs[1].legend()  # Add legend
        axs[1].set_xlabel(feature_)  # Add x label
        axs[1].set_ylabel('Density')  # Add y label

        # save the plot
        plt.savefig(f'Analysis Plots/Distribution Plots/{feature_}.png')

        plt.show()  # Show the plot

    if ks_p < ALPHA:  # If the p-value is less than 0.05
        print(f"{feature_} is not normally distributed (pv = {ks_p:.5f})")
    else:
        print(f"{feature_} is normally distributed (pv = {ks_p:.5f})")

    if ks_p_uniform < ALPHA:  # If the p-value is less than 0.05
        print(f"{feature_} is not uniformly distributed (pv = {ks_p_uniform:.5f})")
    else:
        print(f"{feature_} is uniformly distributed (pv = {ks_p_uniform:.5f})")

    return ks_stat, ks_p, ks_stat_uniform, ks_p_uniform


def check_dependence(data: pd.DataFrame, feature1: str, **kwargs):
    """
    Checks the dependence of a feature and the other features in the data.
    :param data: the data to check
    :param feature1: the first feature to check
    :param kwargs: the keyword arguments
    :return:
    """

    # Constants
    ALPHA = kwargs.get('alpha', 0.05)  # Get the alpha value
    print_results = kwargs.get('print_results', True)  # Get the print_results value

    # Check the dependence of the feature on the target variable
    chi_stats = {}
    p_vals = {}
    dependence = {}
    for feature2 in data.columns:
        if feature2 != feature1:
            # Perform the Chi-Square test to check the dependence of the feature on the target variable
            chi_stats[feature2], p_vals[feature2], _, _ = (
                stats.chi2_contingency(pd.crosstab(data[feature1], data[feature2])))

            if p_vals[feature2] < ALPHA:
                dependence[feature2] = False
            else:
                dependence[feature2] = True

            # Print the results
            if print_results:
                print(f'Chi-Square Test for {feature1} vs {feature2}:')
                print(f'Chi-Square Statistic: {chi_stats[feature2]:.5f}')
                print(f'P-Value: {p_vals[feature2]:.5f}')

    # return all the chi stats and p values as a dataframe

    return pd.DataFrame({'Chi-Square Statistic': chi_stats, 'P-Value': p_vals,
                         'Dependent': dependence})


def test_theory_feature_improvement(data: pd.DataFrame, feature: str, **kwargs):
    """
    Args:
    data: the data to check
    feature: the feature to check
    kwargs: the keyword arguments
    Returns:
    the results of the model with and without the feature.
    """

    # Constants
    ALPHA = kwargs.get('alpha', 0.05)  # Get the alpha value
    print_results = kwargs.get('print_results', True)  # Get the print_results value
    model_name = kwargs.get('model_name', 'xgb')  # Get the model name
    plot_results = kwargs.get('plot', True)  # Get the plot_results value

    assert model_name in ['xgb', 'nn'], "model_name must be either 'xgb' or 'nn'"

    """
    Step 1: divide the data into train and test sets, and keep a copy of the data without the feature.
    """
    # Create a copy of the data without the feature
    X = data.iloc[:, :-1].copy()
    y = data.iloc[:, -1].copy()  # Get the target variable

    # Create a copy of the data without the feature
    X_no_feature = X.copy()
    X_no_feature.drop(columns=[feature], inplace=True)

    shuffle_state = random.choice(range(1000))  # Get a random shuffle state

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=shuffle_state)

    x_train_no_feature, x_test_no_feature, _, _ = train_test_split(X_no_feature, y, test_size=0.2,
                                                                   random_state=shuffle_state)

    """
    Step 2: train the model with and without the feature
    """
    if model_name == 'xgb':
        from xgboost import XGBClassifier

        # Train the model with the feature
        model_with_feature = XGBClassifier(use_label_encoder=False)
        model_with_feature.fit(x_train, y_train)

        # Train the model without the feature
        model_without_feature = XGBClassifier(use_label_encoder=False)
        model_without_feature.fit(x_train_no_feature, y_train)

    else:
        from sklearn.neural_network import MLPClassifier

        # Train the model with the feature
        model_with_feature = MLPClassifier()
        model_with_feature.fit(x_train, y_train)

        # Train the model without the feature
        model_without_feature = MLPClassifier()
        model_without_feature.fit(x_train_no_feature, y_train)

    """
    Step 3: evaluate the models
    """

    # divide the test data (300 samples) into 10 batches of 30, and calculate the mean accuracy of each batch.
    batch_size = 30
    n_batches = 10
    accuracies_with_feature = []
    accuracies_without_feature = []

    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size

        # Evaluate the model with the feature
        score_with = model_with_feature.score(x_test.iloc[start:end], y_test.iloc[start:end])
        accuracies_with_feature.append(score_with)

        # Evaluate the model without the feature
        score_without = model_without_feature.score(x_test_no_feature.iloc[start:end], y_test.iloc[start:end])
        accuracies_without_feature.append(score_without)

    if plot_results:
        # Plot the accuracies as a curve
        plt.figure()
        accs_with = np.array(accuracies_with_feature)
        accs_without = np.array(accuracies_without_feature)
        x_vals = np.linspace(0, n_batches - 1, 300)
        spl = make_interp_spline(range(n_batches), accs_with, k=3)
        accs_with = spl(x_vals)
        sns.lineplot(x=x_vals, y=accs_with, color=BLUE, label='With Feature')

        spl = make_interp_spline(range(n_batches), accs_without, k=3)
        accs_without = spl(x_vals)
        sns.lineplot(x=x_vals, y=accs_without, color=RED, label='Without Feature')
        plt.xlabel('Batch number')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.title(f"{'XGB' if model_name == 'xgb' else 'NN'} Accuracy with and without '{feature}\n"
                  f"Mean Accuracy with '{feature}': {np.mean(accs_with):.3f}\n"
                  f"Mean Accuracy without '{feature}': {np.mean(accs_without):.3f}")
        try:
            plt.savefig(f'Analysis Plots/Accuracy Plots/{model_name}_{feature}_accuracy.png')
        except FileNotFoundError:
            import os
            os.mkdir('Analysis Plots/Accuracy Plots')
        plt.savefig(f'Analysis Plots/Accuracy Plots/{model_name}_{feature}_accuracy.png')
        plt.show()

    """
    Now we have 2 vectors of 10 accuracies each, one for the model with the feature and one for the model without 
    the feature. Now we apply the statistical analysis to check if the feature improves the accuracy.
    """

    # Shapiro-Wilk test for normality of the differences
    _, p_shapiro = stats.shapiro(np.array(accuracies_with_feature) - np.array(accuracies_without_feature))
    print(f"Normality test for '{feature}' (Shapiro-Wilk):")
    print(f"p-value: {p_shapiro:.5f}", end='')

    if p_shapiro > ALPHA:
        print(f" The differences are normally distributed for '{feature}'.")

        # Perform the paired t-test
        t_stat, p_val = stats.ttest_rel(accuracies_with_feature, accuracies_without_feature, alternative='greater')

        # Print the results
        if print_results:
            print(f"Paired T-Test for '{feature}':")
            print(f"T-Statistic: {t_stat:.5f}")
            print(f"P-Value: {p_val:.5f}")
            if p_val < ALPHA:
                print(f"The feature '{feature}' does not improve the accuracy of the model.")
            else:
                print(f"The feature '{feature}' improves the accuracy of the model.")

    else:
        print(f" The differences are not normally distributed for '{feature}'.")

        # Perform the Wilcoxon signed-rank test
        diff = np.array(accuracies_with_feature) - np.array(accuracies_without_feature)
        t_stat, p_val = stats.wilcoxon(diff, alternative='greater')

        # Print the results
        if print_results:
            print(f"Wilcoxon Signed-Rank Test for '{feature}':")
            print(f"Statistic: {t_stat:.5f}")
            print(f"P-Value: {p_val:.5f}")
            if p_val < ALPHA:  # If the p-value is less than 0.05
                print(f"The feature '{feature}' does not improve the accuracy of the model.")
            else:
                print(f"The feature '{feature}' improves the accuracy of the model.")

    return accuracies_with_feature, accuracies_without_feature


def test_theory_contribution(differences_per_feature: List, **kwargs):
    """
    Args:
    differences_per_feature: List of lists of accuracy difference with and without the feature.
    kwargs: the keyword arguments

    This method performs the ANOVA test to check if the differences in accuracy are significant.

    Returns:
    The results of the ANOVA test.
    """

    # Constants
    ALPHA = kwargs.get('alpha', 0.05)  # Get the alpha value
    print_results = kwargs.get('print_results', True)  # Get the print_results value

    """
    We know that each sample is normally distributed, we need to check for equality of variances.
    We'll use the Levene test for this.
    """

    # Perform the Levene test
    stat, p_val = stats.levene(*differences_per_feature)

    # Print the results
    if print_results:
        print("Levene Test for Equality of Variances:")
        print(f"Statistic: {stat:.5f}")
        print(f"P-Value: {p_val:.5f}")
        if p_val < ALPHA:
            print("The variances are not equal.")
        else:
            print("The variances are equal.")

    # Perform the ANOVA test if the variances are equal
    if p_val > ALPHA:
        f_stat, p_val = stats.f_oneway(*differences_per_feature)

        # Print the results
        if print_results:
            print("ANOVA Test:")
            print(f"F-Statistic: {f_stat:.5f}")
            print(f"P-Value: {p_val:.5f}")
            if p_val < ALPHA:  # If the p-value is less than 0.05 can't reject the null hypothesis
                print("The features contributed equally.")
            else:
                print("The features did not contribute equally.")
    else:
        print("Cannot perform ANOVA test because the variances are not equal.")
        # Perform the Kruskal-Wallis test if the variances are not equal
        stat, p_val = stats.kruskal(*differences_per_feature)

        # Print the results
        if print_results:
            print("Kruskal-Wallis Test:")
            print(f"Statistic: {stat:.5f}")
            print(f"P-Value: {p_val:.5f}")
            if p_val < ALPHA:
                print("The features contributed equally.")
            else:
                print("The features did not contribute equally.")


def plot_feature_biases(data: pd.DataFrame, feature: str, show_rates: bool = True):
    """
    Args:
        data: the data to check
        feature: the feature to check
        show_rates: whether to show the relative acceptance and rejection rates on the plot.

    This method plots the biases of the model with and without the feature.

    Returns:
    None
    """

    # for each value in the feature, find how many times the label was accepted and rejected
    feature_values = data[feature].unique()
    acceptances = []
    rejections = []
    for val in feature_values:
        acceptances.append(data[(data[feature] == val) & (data['HiringDecision'] == 1)].shape[0])
        rejections.append(data[(data[feature] == val) & (data['HiringDecision'] == 0)].shape[0])

    # find the relative acceptances and rejections for each unique value of the feature
    relative_acceptances = [acceptances[i] / (acceptances[i] + rejections[i]) for i in range(len(acceptances))]
    relative_rejections = [rejections[i] / (acceptances[i] + rejections[i]) for i in range(len(rejections))]
    # plot the biases, and print the relative acceptance and rejections on the plot.
    plt.figure()
    plt.bar(feature_values, acceptances, color=BLUE, label='Acceptances')
    plt.bar(feature_values, rejections, color=RED, label='Rejections', bottom=acceptances)
    # add the text to the plot
    if show_rates:
        for i, val in enumerate(feature_values):
            plt.text(val, acceptances[i] / 2, f'{relative_acceptances[i]:.2f}', ha='center', va='center')
            plt.text(val, acceptances[i] + rejections[i] / 2, f'{relative_rejections[i]:.2f}',
                     ha='center', va='center')
    plt.xlabel(feature)
    plt.ylabel('Relative Frequency')
    plt.title(f"Biases in Hiring Decision given '{feature}'")
    plt.legend()

    # save the figure
    try:
        plt.savefig(f'Analysis Plots/Bias Plots/{feature}_biases.png')
    except OSError:
        import os
        os.mkdir('Analysis Plots/Bias Plots')
        plt.savefig(f'Analysis Plots/Bias Plots/{feature}_biases.png')
    plt.show()
