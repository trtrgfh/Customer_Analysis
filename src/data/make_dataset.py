import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df):
  X = df.iloc[:, 1:]
  y = df[["Y"]]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
  return X_train, X_test, y_train, y_test
  
url = "https://raw.githubusercontent.com/trtrgfh/nQvq9cIZQdwmNjk7/main/data/ACME-HappinessSurvey2020.csv"
df = pd.read_csv(url)
X_train, X_test, y_train, y_test = split_data(df)

# Feature Selection
print("The corrlelation between the features Xi and target Y")
print(df.corr().iloc[0])
df2 = df.drop(["X2", "X4"], axis=1)
X_train2, X_test2, y_train2, y_test2 = split_data(df2)
