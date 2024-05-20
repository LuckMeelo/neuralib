##
# LuckMeel, 2023
# neuralib
# File description:
# __init__
##

import numpy as np
import pandas as pd
from typing import Tuple, List


def load_csv(file_path: str, target_columns: List[str] | str=[]) -> Tuple[np.ndarray, np.ndarray]:
    # Reading the CSV file
    df = pd.read_csv(file_path)

    if isinstance(target_columns, str):
        target_columns = [target_columns]

    if not target_columns:
        return (df.values)

    # Seperating features and target
    X = df.drop(target_columns, axis=1)
    y = df[target_columns]

    return (X.values, y.values)
