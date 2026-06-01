import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def _onehot_impute(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Impute numeric columns with median
    X_num = X[num_cols].copy()
    if not X_num.empty:
        X_num = X_num.fillna(X_num.median())

    # Fill categorical missing and one-hot encode
    if cat_cols:
        X_cat = X[cat_cols].fillna("Missing")
        X_cat = pd.get_dummies(X_cat, drop_first=True)
    else:
        X_cat = pd.DataFrame(index=X.index)

    X_pre = pd.concat([X_num, X_cat], axis=1)
    return X_pre


def select_features(X: pd.DataFrame, y, method: str = "mutual_info", k: int = 10, problem: str = "classification", random_state: int = 42):
    """Select top-k features from X given target y.

    Parameters
    - X: DataFrame of predictors
    - y: target array-like (required for score-based methods)
    - method: one of 'mutual_info', 'chi2', 'rf'
    - k: number of features to select
    - problem: 'classification' or 'regression'

    Returns
    - X_selected: DataFrame with selected features (preprocessed dummies)
    - selector: fitted selector object (SelectKBest or SelectFromModel). For 'chi2' returns (SelectKBest, scaler)
    """
    if method not in ("mutual_info", "chi2", "rf"):
        raise ValueError("method must be one of 'mutual_info', 'chi2', 'rf'")

    if method in ("mutual_info", "chi2") and y is None:
        raise ValueError("y (target) is required for score-based selection methods")

    X_pre = _onehot_impute(X)

    if method == "mutual_info":
        score = mutual_info_classif if problem == "classification" else mutual_info_regression
        k = min(k, X_pre.shape[1])
        skb = SelectKBest(score, k=k)
        skb.fit(X_pre.values, y)
        mask = skb.get_support()
        selected_cols = X_pre.columns[mask].tolist()
        return X_pre[selected_cols], skb

    if method == "chi2":
        # chi2 requires non-negative values; scale to [0,1]
        k = min(k, X_pre.shape[1])
        scaler = MinMaxScaler()
        X_pos = scaler.fit_transform(X_pre.fillna(0).values)
        skb = SelectKBest(chi2, k=k)
        skb.fit(X_pos, y)
        mask = skb.get_support()
        selected_cols = X_pre.columns[mask].tolist()
        return X_pre[selected_cols], (skb, scaler)

    # method == 'rf'
    if problem == "classification":
        model = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)

    # SelectFromModel will pick features up to k when using threshold=None and max_features
    selector = SelectFromModel(model, max_features=k, threshold=None)
    selector.fit(X_pre.fillna(0).values, y)
    mask = selector.get_support()
    selected_cols = X_pre.columns[mask].tolist()
    return X_pre[selected_cols], selector


def scale_features(X: pd.DataFrame, scaler: str = "standard", columns: list = None, return_scaler: bool = False):
    """Scale numeric columns of X.

    Parameters
    - X: DataFrame
    - scaler: 'standard' | 'minmax' | 'robust'
    - columns: list of columns to scale (defaults to numeric columns)
    - return_scaler: if True, also return the fitted scaler object

    Returns
    - X_scaled (and scaler if requested)
    """
    X = X.copy()
    if columns is None:
        cols = X.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = columns

    if not cols:
        if return_scaler:
            return X, None
        return X

    if scaler == "standard":
        s = StandardScaler()
    elif scaler == "minmax":
        s = MinMaxScaler()
    elif scaler == "robust":
        s = RobustScaler()
    else:
        raise ValueError("scaler must be one of 'standard','minmax','robust'")

    X[cols] = s.fit_transform(X[cols].fillna(0))

    if return_scaler:
        return X, s
    return X


if __name__ == "__main__":
    # Quick demo when run directly
    try:
        df = pd.read_csv("train.csv")
    except Exception:
        print("train.csv not found in cwd — place Titanic train.csv here to run the demo")
    else:
        # example usage
        y = df["Survived"]
        X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
        X_sel, sel = select_features(X, y, method="mutual_info", k=10)
        X_scaled, scaler = scale_features(X_sel, scaler="standard", return_scaler=True)
        print("Selected columns:", X_sel.columns.tolist())
        print("Scaled sample:")
        print(X_scaled.head())
