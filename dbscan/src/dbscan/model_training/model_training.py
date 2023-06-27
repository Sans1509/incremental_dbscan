import logging
import pickle

from sklearn.cluster import DBSCAN

from src.utils.constant import saved_model


def model_training(data_frame):
    """
    training the model using dbscan
    :param data_frame
    :return model
    """
    X_principal = data_frame[1]
    dbscan = DBSCAN(eps=6.5, min_samples=1000, leaf_size=30, p=2)
    model = dbscan

    # fit initial model
    initial_data = X_principal[0:10000]
    model = model.fit(initial_data)

    with open(saved_model / "dbscan_model.pkl", 'wb') as file:
        logging.info("model is trained successfully")
        pickle.dump(dbscan, file)

    return model
