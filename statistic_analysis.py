import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA

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
        plt.title('PCA 2D')  # Add title

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
        plt.title('PCA 3D')  # Add title

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
        axs[0].grid()  # Add grid

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
        axs[1].grid()  # Add grid

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
