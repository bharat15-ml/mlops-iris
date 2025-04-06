import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Create Data Card
dataset_name = "Iris Flower Dataset"
description = "Famous multiclass classification dataset introduced by Ronald Fisher in 1936."
source = "UCI Machine Learning Repository"
num_samples = df.shape[0]
features = ', '.join(df.columns[:-1])
target_classes = ', '.join(df['species'].unique())
feature_summary = df.describe().to_string()
sample_data = df.sample(5).to_string()

data_card = f"""
 DATA CARD: {dataset_name}
==================================================
 Description: {description}
 Source: {source}
 Number of Samples: {num_samples}
 Features: {features}
 Target Classes: {target_classes}

 Feature Summary:
{feature_summary}

 Sample Data:
{sample_data}
"""

# Save data card to file
with open("../iris_data_card.txt", "w", encoding="utf-8") as f:
    f.write(data_card)

# Save visualization as image
sns.set(style="ticks")
pairplot_fig = sns.pairplot(df, hue='species')
pairplot_fig.fig.suptitle("Iris Dataset Pairplot", y=1.02)
pairplot_fig.savefig("../iris_pairplot.png", dpi=300, bbox_inches='tight')

print("Data card saved to iris_data_card.txt")
print("Pairplot saved to iris_pairplot.png")

