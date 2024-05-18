import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from ta import add_all_ta_features

yf.pdr_override()


class TimeSeriesDataset:
    def __init__(
        self, stock_name, time_steps, scaler, start_date, end_date, keep_ratio=0.8
    ):
        self.time_steps = time_steps
        self.df = pdr.get_data_yahoo([stock_name], start=start_date, end=end_date)
        self.scaler = scaler
        self.labels = self.df["Adj Close"]
        self.time_index = self.df.index
        self.keep_ratio = keep_ratio

        self._process_df()
        self._preprocess()

    def _process_df(self):
        self.df = add_all_ta_features(
            self.df,
            close="Adj Close",
            high="High",
            low="Low",
            open="Open",
            volume="Volume",
        )
        self.df.dropna(axis=1, inplace=True)

        # Drop the columns from the data frame
        redundant_columns = pd.Index(["Open", "High", "Low", "Volume", "Close"])
        self.df = self.df.drop(columns=redundant_columns)

        # choose elements based on variance
        num_columns_to_drop = int(self.df.shape[1] * (1 - self.keep_ratio))
        variances = self.df.var()
        sorted_variances = variances.sort_values()
        columns_to_drop_by_var = sorted_variances[:num_columns_to_drop].index

        self.df = self.df.drop(columns=columns_to_drop_by_var)

        self.n_features = self.df.shape[1]

    def _preprocess(self):
        # Scale dataset
        self.X = []
        for col in self.df.columns:
            self.df[col] = self.scaler.fit_transform(self.df[col].values.reshape(-1, 1))
            # Prepare training set
            scaled_data = self.df[col].values
            X = [
                scaled_data[j - self.time_steps : j]
                for j in range(self.time_steps, len(scaled_data))
            ]
            X = np.reshape(np.array(X), (-1, self.time_steps, 1))
            self.X.append(X)

        # apply scaler to the 3D array
        self.X = np.concatenate(self.X, axis=2)
        self.X = np.reshape(
            self.X, (self.X.shape[0] * self.X.shape[1], self.X.shape[2])
        )  # with shape (n_samples * time_steps, n_features)
        self.X = self.scaler.fit_transform(self.X)
        self.X = np.reshape(
            self.X,
            (self.X.shape[0] // self.time_steps, self.time_steps, self.X.shape[1]),
        )

        scaled_labels = self.scaler.fit_transform(self.labels.values.reshape(-1, 1))
        y = [scaled_labels[i] for i in range(self.time_steps, len(scaled_labels))]
        self.y = np.array(y)
