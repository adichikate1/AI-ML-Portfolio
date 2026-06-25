from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("static/data/tested.csv")
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
print(df["Title"].unique())