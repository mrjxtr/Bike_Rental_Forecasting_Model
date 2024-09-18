import os
import sys
from math import sqrt

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import utility.plots_cfg as plt_c
import utility.plots_save as plt_s

dotenv.load_dotenv()
plt_c.load_cfg()

script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, "../data/raw/daily-bike-share.csv")

df = pd.read_csv(data_path)

X = df[
    [
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
    ]
]

y = df["rentals"]

X[["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]] = X[
    ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
].astype("category")

X = pd.get_dummies(
    X,
    columns=["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"],
    dtype=int,
)

X.info()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


test_score = model.score(X_test, y_test)
r2 = r2_score(y_pred, y_test)
rmse = sqrt(mean_squared_error(y_pred, y_test))

print(test_score)
print(r2)
print(rmse)

# plt_s.export_figs()
