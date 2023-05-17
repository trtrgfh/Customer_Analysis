
url = "https://raw.githubusercontent.com/trtrgfh/nQvq9cIZQdwmNjk7/main/data/ACME-HappinessSurvey2020.csv"
df = pd.read_csv(url)
df.head()

print(f"df shape: {df.shape}")
X = df.iloc[:, 1:]
y = df[["Y"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("The corrlelation between the features Xi and target Y")
df.corr().iloc[0]

df2 = df.drop(["X2", "X4"], axis=1)
# df2 = df.drop(["X2", "X3"], axis=1)

X2 = df2.iloc[:, 1:]
y2 = df2[["Y"]]
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.15, random_state=42)
print(f"X_train shape: {X_train2.shape}")
print(f"X_test shape: {X_test2.shape}")
print(f"y_train shape: {y_train2.shape}")
print(f"y_test shape: {y_test2.shape}")
