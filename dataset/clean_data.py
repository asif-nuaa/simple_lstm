import os

import pandas as pd
from datetime import datetime


def clean_data():
    file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data.csv")
    pd_frame = pd.read_csv(file_name, delimiter=",", index_col=False)

    pd_frame.drop(labels=pd_frame.columns[[0, 9]], inplace=True, axis=1)
    pd_frame.dropna(axis=0, inplace=True)

    cols = pd_frame.columns.tolist()
    target_index = 4
    new_cols = [cols[i] for i in range(len(cols)) if i != target_index]
    new_cols.extend([cols[target_index]])
    pd_frame = pd_frame[new_cols]

    dates = pd_frame.values[:, 0:4].astype(int)
    dates = [datetime(y,m,d,h) for y,m,d,h in dates]

    pd_frame.drop(labels=pd_frame.columns[0:4], inplace=True, axis=1)
    pd_frame["time"] = dates

    pd_frame.to_csv(file_name.replace(".csv", "_clean.csv"), index=False)


if __name__ == '__main__':
    clean_data()
