from google.cloud import storage
import pandas as pd
from sklearn import linear_model
import numpy as np
import joblib

BUCKET_NAME = 'wagon-data-745-daniel-1'
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'
MODEL_NAME = 'taxifare'
MODEL_VERSION = 'v1'
AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"


def get_data():
    """method to get the training data (or a portion of it) from google cloud bucket"""
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}",
                     nrows=1000)
    return df


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df




if __name__ == '__main__':
    df = get_data()
