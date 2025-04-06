import pandas as pd
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris['data'],columns = iris['feature_names'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.to_csv('iris.csv', sep = ',', index = False)
