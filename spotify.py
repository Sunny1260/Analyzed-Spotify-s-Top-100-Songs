import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the dataset
# Ensure your dataset includes columns: 'tempo', 'energy', 'popularity'
# Update the file path to the absolute path where your dataset is located
df = pd.read_csv(r"D:\\vscode\\project 1\\spotify_top_100.csv")

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Check for missing values
def check_missing_values(data):
    print("Missing Values:\n", data.isnull().sum())

check_missing_values(df)

# Basic statistics of the dataset
def basic_statistics(data):
    print("\nBasic Statistics:\n", data.describe())

basic_statistics(df)

# Visualize the distributions of 'tempo', 'energy', and 'popularity'
def plot_distributions(data, columns):
    for column in columns:
        sns.histplot(data[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

plot_distributions(df, ['tempo', 'energy', 'popularity'])

# Scatter plot to visualize relationships
def scatter_plots(data, x_column, y_column):
    sns.scatterplot(data=data, x=x_column, y=y_column)
    plt.title(f"{x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()

scatter_plots(df, 'tempo', 'popularity')
scatter_plots(df, 'energy', 'popularity')

# Calculate correlations and display them
def calculate_correlation(data, col1, col2):
    correlation, p_value = pearsonr(data[col1], data[col2])
    print(f"Correlation between {col1} and {col2}: {correlation:.2f} (p-value: {p_value:.3f})")

calculate_correlation(df, 'tempo', 'popularity')
calculate_correlation(df, 'energy', 'popularity')

# Heatmap of correlations between multiple features
def correlation_heatmap(data, columns):
    correlation_matrix = data[columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

correlation_heatmap(df, ['tempo', 'energy', 'popularity'])

# Summary
print("\nAnalysis Complete! Use visualizations and correlations to interpret the results.")
