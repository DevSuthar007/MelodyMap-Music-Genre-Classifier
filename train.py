"""Train and compare KNN, Decision Tree, Random Forest and SVM on GTZAN features.

Usage:
    python src/train.py --data data/features_30_sec.csv

Outputs (created if missing):
    models/best_music_genre_model.pkl
    models/feature_columns.pkl
    results/model_comparison.csv
    results/feature_importance.csv
    docs/*.png
"""
import argparse
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path).drop_duplicates(subset="filename")
    X = df.drop(columns=["filename", "length", "label"])
    X = X.replace([np.inf, -np.inf], np.nan)
    y = df["label"]
    print(f"Songs: {len(df)} | Genres: {y.nunique()} | Features: {X.shape[1]}")
    return X, y


def build_searches(cv):
    """Each model: preprocessing pipeline + hyper-parameter grid (scored by macro F1)."""
    imputer = ("imputer", SimpleImputer(strategy="median"))
    scaler = ("scaler", StandardScaler())

    specs = {
        "KNN": (
            Pipeline([imputer, scaler, ("classifier", KNeighborsClassifier())]),
            {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__metric": ["euclidean", "manhattan"],
            },
        ),
        "Decision Tree": (
            Pipeline([imputer, ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [5, 10, 15, None],
                "classifier__min_samples_split": [2, 5, 10],
            },
        ),
        "Random Forest": (
            Pipeline([imputer, ("classifier", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))]),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 20],
                "classifier__min_samples_split": [2, 5],
            },
        ),
        "SVM": (
            Pipeline([imputer, scaler, ("classifier", SVC())]),
            [
                {"classifier__kernel": ["linear"], "classifier__C": [0.1, 1, 10]},
                {
                    "classifier__kernel": ["rbf"],
                    "classifier__C": [0.1, 1, 10],
                    "classifier__gamma": ["scale", "auto"],
                },
            ],
        ),
    }
    return {
        name: GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        for name, (pipe, grid) in specs.items()
    }


def main(args):
    for folder in ("models", "results", "docs"):
        os.makedirs(folder, exist_ok=True)

    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    searches = build_searches(cv)

    models, predictions, rows = {}, {}, []
    for name, search in searches.items():
        print(f"\nTraining {name} ...")
        search.fit(X_train, y_train)
        print("  best params :", search.best_params_)
        print("  best CV F1  :", round(search.best_score_, 4))

        best = search.best_estimator_
        y_pred = best.predict(X_test)
        models[name], predictions[name] = best, y_pred

        rows.append({
            "Algorithm": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
        })
        print(classification_report(y_test, y_pred, zero_division=0))

    results = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    print("\n", results.round(4).to_string(index=False))

    # ---- save best model + results ----
    best_name = results.loc[0, "Algorithm"]
    joblib.dump(models[best_name], "models/best_music_genre_model.pkl")
    joblib.dump(list(X.columns), "models/feature_columns.pkl")
    results.to_csv("results/model_comparison.csv", index=False)
    print(f"\nBest model: {best_name} ({results.loc[0, 'Accuracy']:.1%} test accuracy) -> saved to models/")

    # ---- feature importance (Random Forest) ----
    rf = models["Random Forest"].named_steps["classifier"]
    importance = (pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
                  .sort_values("Importance", ascending=False))
    importance.to_csv("results/feature_importance.csv", index=False)

    # ---- graphs ----
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results.melt(id_vars="Algorithm", var_name="Metric", value_name="Score"),
                x="Algorithm", y="Score", hue="Metric")
    plt.ylim(0, 1)
    plt.title("Performance Comparison of All Classification Algorithms")
    plt.tight_layout()
    plt.savefig("docs/model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    genres = sorted(y.unique())
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for (name, y_pred), ax in zip(predictions.items(), axes.flatten()):
        sns.heatmap(confusion_matrix(y_test, y_pred, labels=genres), annot=True, fmt="d",
                    cmap="Blues", xticklabels=genres, yticklabels=genres, ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted Genre")
        ax.set_ylabel("Actual Genre")
    plt.tight_layout()
    plt.savefig("docs/confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 7))
    sns.barplot(data=importance.head(15), x="Importance", y="Feature")
    plt.title("Top 15 Important Audio Features (Random Forest)")
    plt.tight_layout()
    plt.savefig("docs/feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/features_30_sec.csv",
                        help="path to GTZAN features_30_sec.csv")
    main(parser.parse_args())
