import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.impute import SimpleImputer


def train_random_forest_classifier(
    features_df: pd.DataFrame,
    random_state: int = 42,
    real_world_weights: dict | None = None,
    encode_categorical: bool = True,
    param_grid: dict | None = None,
    n_cv_folds: int = 5,
):
    """
    Train a Random Forest classifier with household-level splitting,
    optional preprocessing, and cross-validated hyperparameter tuning.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must contain 'label' and 'ean_id'
    random_state : int
        Random seed
    real_world_weights : dict, optional
        Class proportions for unbalanced test evaluation
    encode_categorical : bool
        If False, categorical features are dropped
        Note: the original paper uses numeric features only.
    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV
    n_cv_folds : int
        Number of CV folds (grouped by ean_id)

    Returns
    -------
    Same return objects as the original implementation
    """

    # ======================================================
    # Step 1: Separate X / y and identify feature types
    # ======================================================
    X = features_df.drop(columns="label")
    y = features_df["label"]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if "ean_id" in numeric_features:
        numeric_features.remove("ean_id")
    if "ean_id" in categorical_features:
        categorical_features.remove("ean_id")

    # ======================================================
    # Step 2: Household-level train/test split (balanced)
    # ======================================================
    households = features_df[["ean_id", "label"]].drop_duplicates()

    train_hh, test_hh_bal = train_test_split(
        households,
        test_size=0.3,
        stratify=households["label"],
        random_state=random_state,
    )

    train_idx = features_df["ean_id"].isin(train_hh["ean_id"])
    test_idx_bal = features_df["ean_id"].isin(test_hh_bal["ean_id"])

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test_bal, y_test_bal = X.loc[test_idx_bal], y.loc[test_idx_bal]

    # ======================================================
    # Step 2b: Optional unbalanced test set
    # ======================================================
    X_test_unbal = y_test_unbal = y_pred_unbal = None

    if real_world_weights is not None:
        test_hh_unbal = []
        n_test = len(test_hh_bal)

        for cls, weight in real_world_weights.items():
            cls_hh = households[households["label"] == cls]
            n_draw = max(1, int(round(n_test * weight)))
            test_hh_unbal.append(
                cls_hh.sample(n=n_draw, replace=True, random_state=random_state)
            )

        test_hh_unbal = pd.concat(test_hh_unbal)
        idx_unbal = features_df["ean_id"].isin(test_hh_unbal["ean_id"])
        X_test_unbal, y_test_unbal = X.loc[idx_unbal], y.loc[idx_unbal]

    # ======================================================
    # Step 3: Preprocessing pipeline (configurable)
    # ======================================================
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])
        transformers.append(("num", numeric_pipeline, numeric_features))
        
    if categorical_features and encode_categorical:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # ======================================================
    # Step 4: Model and hyperparameter grid
    # ======================================================
    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
    )

    if param_grid is None:
        param_grid = {
            "rf__n_estimators": [200, 400],
            "rf__max_depth": [None, 10, 20],
            "rf__min_samples_split": [2, 5],
            "rf__min_samples_leaf": [1, 2],
            "rf__max_features": ["sqrt", "log2"],
        }

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("rf", rf),
        ]
    )

    # ======================================================
    # Step 5: Cross-validated hyperparameter tuning
    # ======================================================
    groups = features_df.loc[train_idx, "ean_id"]
    cv = GroupKFold(n_splits=n_cv_folds)

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=3,
    )

    grid.fit(X_train, y_train, groups=groups)

    pipeline = grid.best_estimator_

    # ======================================================
    # Step 6: Predictions and evaluation
    # ======================================================
    y_pred_bal = pipeline.predict(X_test_bal)

    print("\nBalanced test set evaluation:")
    print(classification_report(y_test_bal, y_pred_bal))
    report_bal = classification_report(y_test_bal, y_pred_bal, output_dict=True)
    report_bal_df = pd.DataFrame(report_bal).T


    if X_test_unbal is not None:
        y_pred_unbal = pipeline.predict(X_test_unbal)
        print("\nUnbalanced (real-world) test set evaluation:")
        print(classification_report(y_test_unbal, y_pred_unbal))
        report_unbal = classification_report(y_test_unbal, y_pred_unbal, output_dict=True)
        report_unbal_df = pd.DataFrame(report_unbal).T

    # ======================================================
    # Step 7: Feature importances
    # ======================================================
    rf_fitted = pipeline.named_steps["rf"]
    importances = rf_fitted.feature_importances_
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    feature_importances = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # ======================================================
    # Step 7b: Top 10 transformed features
    # ======================================================
    X_transformed = pipeline.named_steps["preprocessor"].transform(X)
    X_transformed_df = pd.DataFrame(
        X_transformed, columns=feature_names, index=X.index
    )

    top10_features = feature_importances["feature"].head(10)
    top10_df = X_transformed_df[top10_features].copy()
    top10_df["label"] = y
    top10_df["ean_id"] = features_df["ean_id"]

    # ======================================================
    # Step 8: Confusion matrices
    # ======================================================
    def plot_confusion(y_true, y_pred, title):
        labels = np.unique(y)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Predicted label",
            ylabel="True label",
            title=title,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if i == j else "black"
                ax.text(j, i, cm[i, j], ha="center", va="center", color=color)

        fig.tight_layout()
        return fig, ax

    fig_cm, ax_cm = plot_confusion(
        y_test_bal, y_pred_bal, "Confusion Matrix (Balanced Test Set)"
    )

    fig_cm_unbal = ax_cm_unbal = None
    if y_test_unbal is not None:
        fig_cm_unbal, ax_cm_unbal = plot_confusion(
            y_test_unbal, y_pred_unbal, "Confusion Matrix (Unbalanced Test Set)"
        )

    # ======================================================
    # Step 9: Feature importance plot
    # ======================================================
    top_k = feature_importances.head(10).iloc[::-1]
    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
    ax_imp.barh(top_k["feature"], top_k["importance"])
    ax_imp.set_title("Top 10 Feature Importances (Random Forest)")
    ax_imp.set_xlabel("Importance")
    fig_imp.tight_layout()

    return (
        pipeline,
        X_test_bal,
        y_test_bal,
        y_pred_bal,
        X_test_unbal,
        y_test_unbal,
        y_pred_unbal,
        feature_importances,
        top10_df,
        report_bal_df,
        report_unbal_df,
        (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal),
    )
