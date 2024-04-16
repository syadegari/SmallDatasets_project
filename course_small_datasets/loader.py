import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from numpy import ndarray
from typing import Union, Tuple


class DataBuilder(Dataset):
    def __init__(self, data: Union[str, pd.DataFrame], train=True):
        super().__init__()
        x_train, x_test, self.scaler = load_and_standardize_data(data)
        if train:
            self.x = torch.from_numpy(x_train)
        else:
            self.x = torch.from_numpy(x_test)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.shape[0]


def load_and_standardize_data(data: Union[str, pd.DataFrame]) -> Tuple[ndarray, ndarray, StandardScaler]:
    """
    Parameters:
    - path (Union[str, pd.DataFrame]): The file path to a CSV file or a dataframe.

    Returns:
    - X_train (ndarray): The standardized train data.
    - X_test (ndarray): The standardized test data.
    - scalar (StandardScaler): The scaler used for data standardization of the training set.
    """
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise ValueError("Data should be either a file path or a DataFrame.")

    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype("float32")
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=101)
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    return X_train, X_test, scalar
