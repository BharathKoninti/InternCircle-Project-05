import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["pclass", "sex", "age", "fare"]

df = sns.load_dataset("titanic")[["survived", *FEATURES]].dropna().copy()
df["sex"] = (df["sex"] == "female").astype(int)

X = df[FEATURES]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model.joblib")
print("Saved model.joblib")
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
