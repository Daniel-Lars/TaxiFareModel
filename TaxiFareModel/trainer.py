# imports
from memoized_property import memoized_property
import mlflow
import joblib
from mlflow.tracking import MlflowClient
from TaxiFareModel.data import clean_data, get_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from google.cloud import storage

from TaxiFareModel.params import MLFLOW_URI, EXPERIMENT_NAME, BUCKET_NAME, STORAGE_LOCATION


#MLFLOW_URI = "https://mlflow.lewagon.co/"
#EXPERIMENT_NAME = "[NL] [AMS] [Daniel]  + V1"
#BUCKET_NAME = 'wagon-data-745-daniel-1'
#STORAGE_LOCATION = 'models/TaxiFareModel/model.joblib'


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])

        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preproc_pipe = ColumnTransformer([
        ('distance', dist_pipe, [
            "pickup_latitude",
            "pickup_longitude",
            'dropoff_latitude',
            'dropoff_longitude'
        ]),
        ('time', time_pipe, ['pickup_datetime'])],
        remainder="drop")

        self.pipeline = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', LinearRegression())])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.mlflow_log_param('model', 'linear')
        self.pipeline.fit(self.X, self.y )

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred,y_test)
        self.mlflow_log_metric('rmse', rmse)
        return round(rmse,2)


    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)





    def upload_model_to_gcp(self):

        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        blob = bucket.blob(STORAGE_LOCATION)

        blob.upload_from_filename('model.joblib')


    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.pipeline, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        self.upload_model_to_gcp()

        print(
            f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}"
        )


if __name__ == "__main__":

    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df['fare_amount']
    X = df.drop(columns='fare_amount', axis=1)
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
    # train
    trainer = Trainer(X_train,y_train)
    trainer.set_experiment_name(EXPERIMENT_NAME)
    trainer.run()
    rmse = trainer.evaluate(X_test,y_test)
    # evaluate
    print(f"rmse: {rmse}")
    trainer.save_model()
