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
from sklearn.preprocessing import FunctionTransformer

"""
Stacking and naive ensemble classifiers with explicit pipeline construction.

For stacking, base learners are passed as unfitted pipelines so the procedure
can re-train and cross-validate them on each fold internally.
For the naive approach, base learners are passed as pre-fitted pipelines.

In both cases, preprocessing and train/test splitting are handled explicitly
outside the core training functions.
"""

# ==========================================
# ==========================================
# General functions
# ==========================================
# ==========================================

def build_preprocessor(
        column_list, 
        df_all,
        encode_categorical: bool = True,
    ):
    """
    Create preprocessing ColumnTransformer matching the original pipeline.
    
    Similar to preprocessing steps in src.classification_random_forest_CV.train_random_forest_classifier
    but extracted as a standalone function to allow separate preprocessing.
    Used in both meta-model approaches:
    - Approach 1: Preprocess full dataset before extracting top-k features from each model
    - Approach 2: Create preprocessing step for stacking pipelines
        
    Parameters
    ----------
    column_list : list of str
        Column names to preprocess
    df_all : pd.DataFrame
        Full dataframe (used to infer column dtypes)
    encode_categorical : bool
        If False, categorical features are dropped
        Note: the original paper uses numeric features only.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Preprocessor with StandardScaler + SimpleImputer for numeric features
        and OneHotEncoder + SimpleImputer for categorical features.
        'ean_id' is excluded from preprocessing.
    """
    numeric_features = [col for col in column_list if df_all[col].dtype in ['int64', 'float64']]
    categorical_features = [col for col in column_list if df_all[col].dtype in ['object', 'category']]

    for col in ["ean_id"]:
        if col in numeric_features:
            numeric_features.remove(col)
        if col in categorical_features:
            categorical_features.remove(col)

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

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return preprocessor

def household_train_test_split(df, test_size = 0.3, random_state = 42, real_world_weights = None, precomputed_split = None):
    """
    Split households into train/test sets and return the corresponding datasets.

    Splits at household level to prevent data leakage. 
    
    Optionally accepts a precomputed split to ensure consistency across 
    multiple binary base learner models.

    Used for both meta-model approaches. 
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset containing 'ean_id', 'label', and features
    test_size : float, default=0.3
        Proportion of households reserved for the balanced test set.
    random_state : int, default=42
        Random seed for reproducibility
    real_world_weights : dict, optional
        Class proportions for unbalanced test set (keys=labels, values=weights)
        If None, only balanced test set is created
    precomputed_split : dict, optional
        Pre-computed split with keys 'train_hh' and 'test_hh_bal'.
        If None, a new split is performed.
        Used for pooling binary learners. 
    
    Returns
    -------
    train_idx : pd.Series (bool)
        Boolean mask for training observations
    test_idx_bal : pd.Series (bool)
        Boolean mask for balanced test observations
    X_train : pd.DataFrame
        Training features (excludes 'ean_id' and 'label')
    y_train : pd.Series
        Training labels
    X_test_bal : pd.DataFrame
        Balanced test features
    y_test_bal : pd.Series
        Balanced test labels
    X_test_unbal : pd.DataFrame or None
        Unbalanced (real-world weighted) test features
    y_test_unbal : pd.Series or None
        Unbalanced test labels

    Notes
    -----
    A fixed household-level split is used across all base learners to prevent
    leakage at the meta level: if each base learner split independently, some
    would train on households that others held out, corrupting the meta-learner's
    out-of-sample predictions quality. 
    """

    X = df.drop(columns=["label", "ean_id"])
    y = df["label"]

    households = df[["ean_id", "label"]].drop_duplicates()

    # Reuse the precomputed split if there is one
    if precomputed_split is not None:
        train_hh    = precomputed_split["train_hh"]
        test_hh_bal = precomputed_split["test_hh_bal"]
    else:
        train_hh, test_hh_bal = train_test_split(
            households,
            test_size=test_size,
            stratify=households["label"],
            random_state=random_state,
        )

    # Train, test datasets (balanced):
    train_idx = df["ean_id"].isin(train_hh["ean_id"])
    test_idx_bal = df["ean_id"].isin(test_hh_bal["ean_id"])

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test_bal, y_test_bal = X.loc[test_idx_bal], y.loc[test_idx_bal]

    # Test datasets (unbalanced): 
    X_test_unbal = y_test_unbal = None # Avoids error if real_world_weights is None

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
        idx_unbal = df["ean_id"].isin(test_hh_unbal["ean_id"])
        X_test_unbal, y_test_unbal = X.loc[idx_unbal], y.loc[idx_unbal]

    return train_idx, test_idx_bal, X_train, y_train, X_test_bal, y_test_bal, X_test_unbal, y_test_unbal

def train_random_forest_classifier_presplit(
        features_df, 
        train_idx, 
        X_train, 
        y_train, 
        X_test_bal, 
        y_test_bal, 
        X_test_unbal, 
        y_test_unbal, 
        random_state: int = 42,
        param_grid: dict | None = None, 
        n_cv_folds: int = 5):
    """
    Train Random Forest classifier on already-preprocessed, pre-splitted data.
    
    Similar to src.classification_random_forest_CV.train_random_forest_classifier
    but removes preprocessing steps from the pipeline and accepts pre-split
    train/test datasets as inputs. This allows preprocessing to happen separately.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Full preprocessed dataset (with 'ean_id' and 'label')
    train_idx : pd.Series (bool)
        Boolean mask for training observations
    X_train : pd.DataFrame
        Training features (already preprocessed, excludes 'ean_id' and 'label')
    y_train : pd.Series
        Training labels
    X_test_bal : pd.DataFrame
        Balanced test features (already preprocessed)
    y_test_bal : pd.Series
        Balanced test labels
    X_test_unbal : pd.DataFrame or None
        Unbalanced test features (already preprocessed)
    y_test_unbal : pd.Series or None
        Unbalanced test labels
    random_state : int, default=42
        Random seed for reproducibility
    param_grid : dict, optional
        Hyperparameter grid for GridSearchCV. If None, uses default grid.
    n_cv_folds : int, default=5
        Number of CV folds for GroupKFold
    
    Returns
    -------
    tuple (12 elements)
        pipeline : Pipeline
            Fitted pipeline (contains only 'rf' step, no preprocessing)
        X_test_bal : pd.DataFrame
            Balanced test features (passthrough)
        y_test_bal : pd.Series
            Balanced test labels (passthrough)
        y_pred_bal : np.ndarray
            Predictions on balanced test set
        X_test_unbal : pd.DataFrame or None
            Unbalanced test features (passthrough)
        y_test_unbal : pd.Series or None
            Unbalanced test labels (passthrough)
        y_pred_unbal : np.ndarray or None
            Predictions on unbalanced test set
        feature_importances : pd.DataFrame
            Feature importances sorted by importance
        top10_df : pd.DataFrame
            Top 10 features with 'label' and 'ean_id'
        report_bal_df : pd.DataFrame
            Classification report for balanced test
        report_unbal_df : pd.DataFrame
            Classification report for unbalanced test
        figures : tuple
            (fig_imp, ax_imp, fig_cm, ax_cm, fig_cm_unbal, ax_cm_unbal)

    """

    X = features_df.drop(columns=["label", "ean_id"])
    y = features_df["label"]

    # Initialization: 
    y_pred_unbal = None
    report_unbal_df = None

    # ======================================================
    # Step 1: Model and hyperparameter grid
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

    # Pipeline is just RF classifier: preprocessor is outsourced
    pipeline = Pipeline(
        steps=[
            ("rf", rf),
        ]
    )

    # ======================================================
    # Step 2: Cross-validated hyperparameter tuning
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
    # Step 3: Predictions and evaluation
    # ======================================================
    y_pred_bal = pipeline.predict(X_test_bal)

    print("\n" + "*"*50)
    print("*** Balanced test set evaluation: ***")
    print("*"*50 + "\n")
    print(classification_report(y_test_bal, y_pred_bal))
    report_bal = classification_report(y_test_bal, y_pred_bal, output_dict=True)
    report_bal_df = pd.DataFrame(report_bal).T

    if X_test_unbal is not None:
        y_pred_unbal = pipeline.predict(X_test_unbal)
        print("\n" + "*"*50)
        print("*** Unbalanced (real-world) test set evaluation: ***")
        print("*"*50 + "\n")
        print(classification_report(y_test_unbal, y_pred_unbal))
        report_unbal = classification_report(y_test_unbal, y_pred_unbal, output_dict=True)
        report_unbal_df = pd.DataFrame(report_unbal).T

    # ======================================================
    # Step 4: Feature importances
    # ======================================================
    rf_fitted = pipeline.named_steps["rf"]
    importances = rf_fitted.feature_importances_
    feature_names = X_train.columns.tolist()

    feature_importances = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # ======================================================
    # Step 4b: Top 10 transformed features
    # ======================================================
    top10_features = feature_importances["feature"].head(10).tolist()
    top10_df = X[top10_features].copy()
    top10_df["label"] = y
    top10_df["ean_id"] = features_df["ean_id"]

    # ======================================================
    # Step 5: Confusion matrices
    # ======================================================

    fig_cm, ax_cm = plot_confusion(
        y_test_bal, y_pred_bal, "Confusion Matrix (Balanced Test Set)"
    )

    fig_cm_unbal = ax_cm_unbal = None
    if y_test_unbal is not None:
        fig_cm_unbal, ax_cm_unbal = plot_confusion(
            y_test_unbal, y_pred_unbal, "Confusion Matrix (Unbalanced Test Set)"
        )

    # ======================================================
    # Step 6: Feature importance plot
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

# ==========================================
# ==========================================
# Functions for the naive approach
# ==========================================
# ==========================================

def build_top_k_feature_df(saved_pipelines, df_all, k=10): 
    """
    Extract top-k features from saved models and create preprocessed dataset.
    
    Used for Meta-model Approach 1 (naive feature selection ensemble):
    Selects the most important features from each base model and trains a 
    final RF on the pooled feature set.
    
    Since top-k features are identified from preprocessed feature space
    (e.g., 'num__consumption_mean'), the entire dataset must be preprocessed
    first, then filtered to selected features. This preprocessing step is 
    separate from model fitting to avoid re-preprocessing during training.
    
    Parameters
    ----------
    saved_pipelines : dict
        {Model name:fitted Pipeline with 'preprocessor' and 'rf' steps}
    df_all : pd.DataFrame
        Full dataset with 'ean_id', 'label', and all features
    k : int, default is 10
        Number of top features per model
            
    Returns
    -------
    pd.DataFrame
        Preprocessed top-k features + 'ean_id' + 'label'
        Features renamed with model prefix (e.g., 'sma_feature') to identify
        dominating feature sets. 
    """

    # ======================================================
    # 1. Identify top-k features from each saved base learner
    # ======================================================

    dict_top_k = {}

    for feature_set_name, pipeline in saved_pipelines.items():
        pipeline_fitted = pipeline.named_steps["rf"]
        importances = pipeline_fitted.feature_importances_
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

        feature_importances = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        dict_top_k[feature_set_name] = [feature for feature in feature_importances["feature"].head(k)]

    # Union of top-k features across all base learners
    all_top_features = []

    for feat_list in dict_top_k.values(): 
        for feature in feat_list: 
            all_top_features.append(feature)

    # ======================================================
    # 2. Preprocess entire dataset
    # ======================================================        

    column_list = [col for col in df_all.columns if col not in ["ean_id", "label"]]
    preprocessor = build_preprocessor(column_list, df_all)

    array_all_preprocessed = preprocessor.fit_transform(df_all[column_list])

    # Convert df_all_preprocessed back into a pandas df

    preprocessed_feature_names = preprocessor.get_feature_names_out()

    df_all_preprocessed = pd.DataFrame(
        array_all_preprocessed,
        columns=preprocessed_feature_names,
        index=df_all.index  # Keep same indices
    )

    # ======================================================
    # 3. Filter to keep top-k features + 'ean_id' + 'label'
    # ======================================================

    # Keep columns that are identified as top features 
    columns_to_keep = [c for c in df_all_preprocessed.columns if c in all_top_features]
    df_all_preprocessed = df_all_preprocessed[columns_to_keep].copy()

    # Re-attach ean_id and label (dropped during preprocessing)
    df_all_preprocessed['ean_id'] = df_all['ean_id'].values
    df_all_preprocessed['label'] = df_all['label'].values

    # ======================================================
    # 4. Add model prefixed for interpretability 
    # ======================================================

    rename_dictionary = {}

    for model, features in dict_top_k.items():
        for feature in features: 
            if feature in df_all_preprocessed.columns:
                rename_dictionary[feature] = f"{model}_{feature}"

    df_all_preprocessed.rename(columns=rename_dictionary, inplace=True)

    print(f"\nApproach 1: shape of pre-processed dataset of top {k} features: {df_all_preprocessed.shape}\n")

    return df_all_preprocessed

def plot_confusion(y_true, y_pred, title):
    """
    Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    title : str
        Plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    labels = np.unique(y_true)
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

# ==========================================
# ==========================================
# Functions for the stacking classifier
# ==========================================
# ==========================================

# Pipeline steps: 
# 1. Merge features into one dataset 
#    -> combine_intersect_df
# 2. Build a 'selector' that slices it per feature set 
#    -> build_selector
# 3. Build a 'preprocessor' that cleans NaNs and creates 0/1
#    -> build_preprocessor 
#      (used for the naive approach too, see general functions)

def combine_intersect_df(
        df_list, 
        merge_keys
    ):
    """
    Merge a list of feature DataFrames on their common households.

    Retains only EAN IDs present in all DataFrames (intersection), then
    merges them into a single dataset. Labels are taken from the first
    DataFrame only.

    Parameters
    ----------
    df_list : list of pd.DataFrame
        Feature DataFrames, each containing 'ean_id', 'label', and features.
        The first DataFrame's 'label' column is kept; others are dropped.
    merge_keys : list of str
        Columns to merge on (e.g. ['ean_id', 'moy']).

    Returns
    -------
    pd.DataFrame
        Merged dataset with all features and a single 'label' column.

    Note
    ----
    The StackingClassifier requires a single dataset, so all feature sets must
    share the same households. Since some feature sets may drop households during
    preprocessing (e.g., IFeel), only households present across all feature sets
    are retained.
    """

    # Common households

    common_ean = set(df_list[0]["ean_id"])
    for df in df_list[1:]:
        common_ean &= set(df["ean_id"])

    # Intersection of each df with the common EAN

    starting_df = df_list[0]
    inter_df_list = [starting_df[starting_df["ean_id"].isin(common_ean)].reset_index(drop=True)]
    for df in df_list[1:]: 
        inter_df_list.append(df[df["ean_id"].isin(common_ean)].drop('label', axis=1).reset_index(drop=True))

    # Merge datasets: 

    df_all = inter_df_list[0]
    for df in inter_df_list[1:]:
        df_all = df_all.merge(df, on=merge_keys, how='inner')

    return df_all

def build_selector(
        df, 
        all_features
    ):
    """
    Build a sklearn transformer that selects the columns of df from a larger dataset.

    Used as the first step in each base learner pipeline to slice the relevant
    feature subset from the full merged dataset passed to the StackingClassifier.

    Parameters
    ----------
    df : pd.DataFrame
        Feature set whose columns should be selected (e.g. df_sma)
    all_features : pd.DataFrame
        Full merged dataset containing all feature sets

    Returns
    -------
    FunctionTransformer
        Transformer that filters input to shared columns only.
    """

    shared_columns = [col for col in df.columns if col in all_features.columns]
    df_transformer = FunctionTransformer(lambda K: K[shared_columns], validate = False)

    return df_transformer, shared_columns