import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


mlflow.set_tracking_uri("http://127.0.0.1:5000")


wine = load_wine()
x = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

max_depth = 10
n_estimators = 10


with mlflow.start_run():
    clf = RandomForestClassifier(max_depth = max_depth, n_estimators = n_estimators, random_state = 42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', acc)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    mlflow.set_tags({
        "Author": "vikram",
        "Project": "learning ml-flow"
    })

    mlflow.sklearn.log_model(clf, "random_forest")
    mlflow.log_artifact(__file__)

    print(acc)
