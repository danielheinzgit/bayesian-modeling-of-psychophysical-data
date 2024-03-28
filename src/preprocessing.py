import pandas as pd
import numpy as np


def dataset_1_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df["hit"] = df["hit"].astype(int)
    df["congruent"].replace({True: "Congruent", False: "Incongruent"}, inplace=True)
    return df


def dataset_2_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    return df


def dataset_3_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df["Confidence_discrete"] = df["Confidence"].astype("category")
    return df

def encode_distance_setting(col : pd.Series) -> np.ndarray:
    """
    Encodes the distance setting as numerical values.

    Parameters:
        col: pd.Series
            A Pandas Series containing the distance setting.

    Returns:
        np.ndarray
            An array of numerical values representing the distance setting.
    """
    return col.map({'high' : 1, 'mid' : 2, 'low' : 3}).values

def encode_number_of_objects(col : pd.Series) -> np.ndarray:
    """
    Encodes the number of objects as numerical values.
    
    Parameters:
        col: pd.Series
            A Pandas Series containing the number of objects.
    
    Returns:
        np.ndarray
            An array of numerical values representing the number of objects.
    """
    return col.map({2 : 1, 3 : 2, 4 : 3, 8 : 4, 16 : 5}).values
