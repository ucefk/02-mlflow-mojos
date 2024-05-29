import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow

mlflow.set_tracking_uri("sqlite:///backend_db/mlflow.db")
mlflow.set_experiment("nyc_taxi-expirement")

mlflow.sklearn.autolog(log_datasets = False)



def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    with mlflow.start_run():

        # mlflow.log_param("max_depth", 10)
        # mlflow.log_param("random_state", 0)
        # mlflow.log_param("train_data_path", train_path)
        # mlflow.log_param("val_data_path", val_path)

        # mlflow.set_tag("developer", "Youcef")

        train_path = os.path.join(data_path, "train.pkl")
        val_path = os.path.join(data_path, "val.pkl")

        X_train, y_train = load_pickle(train_path)
        X_val, y_val = load_pickle(val_path)

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        
        # mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    run_train()
