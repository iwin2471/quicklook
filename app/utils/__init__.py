import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster


def get_corr_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)

def get_upper_triangle(corr: pd.DataFrame) -> pd.Series:
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    return corr.where(mask)

def remove_diagonal(corr: pd.DataFrame) -> pd.DataFrame:
    return corr.where(lambda x: ~np.eye(len(x), dtype=bool))
    

def get_possible_pair(df: pd.DataFrame, column) -> pd.DataFrame:
    corr = get_corr_numeric(df)
    return corr[column].sort_values(ascending=False)[1:]


def get_null_columns(df: pd.DataFrame) -> pd.Series:
    null_sums = df.isnull().sum()
    return null_sums[null_sums > 0]

def drop_null_columns(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    null_columns = get_null_columns(df)
    return df.drop(null_columns.index, axis=1, inplace=inplace)

def remove_duplicates(df: pd.DataFrame, inplace: bool = False, reset_index: bool = False) -> pd.DataFrame:
    df.drop_duplicates(inplace=inplace)
    if reset_index:
        df.reset_index(drop=True, inplace=inplace)
    return


def get_elbow_point(df: pd.DataFrame, column: str) -> int:
    scores = []
    for k in range(2, 10):
        labels = fcluster(Z, k, criterion='maxclust')
        score = silhouette_score(data, labels)
        scores.append(score)

    plt.plot(range(2, 10), scores)
    plt.show()




