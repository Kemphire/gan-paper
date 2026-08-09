from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd


class LoggingSMOTE(SMOTE):
    iteration = 1
    def fit_resample(self, X, y):
        print(f"Iteration -> {self.iteration}")
        print(f"SMOTE input: {X.shape}")
        X_res, y_res = super().fit_resample(X, y)
        print(f"SMOTE output: {X_res.shape}")
        print("\n\n")
        LoggingSMOTE.iteration += 1
        return X_res, y_res


def main():

    df = pd.read_csv(f"./Datasets/bcwd.csv")

    df = df.dropna(axis=0)

    print(df.shape)


    X = df.drop(["Class"], axis=1)

    df["Class"] = [
        int(x) for x in df["Class"]
    ]  # in case of float class e.g. (0.0,1.0)

    y = df["Class"]

    # print(y)

    # X_train, X_test, y_train, y_test = train_test_split(df.drop(['Class'], axis=1), df['Class'], test_size=0.2, random_state=10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10, stratify=y
    )

    pipeline = Pipeline(
        [("smote", LoggingSMOTE(random_state=42)), ("clf", RandomForestClassifier())]
    )

    cross_val_predict(estimator=pipeline, X=X_train, y=y_train, cv=5)


main()