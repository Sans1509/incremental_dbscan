import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def pre_processing(data_frame):
    """
        getting the dataframe and pre processing it
        @param data_frame
        @return dataframe
    """
    read_file = pd.read_csv(data_frame, delimiter=";",
                            low_memory=False)
    data_frame = pd.DataFrame(read_file)
    processed_dropped_data = drop_columns(data_frame)
    missing_value_data = handling_missing_values(processed_dropped_data)
    selected_feature_data = feature_selection(missing_value_data)
    scaled_data = scaling(selected_feature_data)
    normalized_data = pd.DataFrame(scaled_data)
    new_reduced_data = dimensionality_reduction(normalized_data)
    batches = create_batches(new_reduced_data, 10000)
    data = [batches, new_reduced_data]

    return data


def dimensionality_reduction(data_frame):
    """
    reducing the dimension of dataframe
    :param data_frame
    :return dataframe
    """
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(data_frame)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    return X_principal


def drop_columns(data_frame):
    """
    getting the dataframe and dropping the un_necessary column
    @param data_frame
    @return data_frame
    """
    dropping_columns = ['Date', 'Time']
    data_frame = data_frame.drop(dropping_columns, axis=1)
    return data_frame


def handling_missing_values(data_frame):
    """
    getting the dataframe and handling missing values in each column
    :param data_frame
    :return data_frame
    """
    columns_to_preprocess = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                             'Sub_metering_1',
                             'Sub_metering_2', 'Sub_metering_3']
    for column in columns_to_preprocess:
        data_frame[column] = data_frame[column].replace('?', np.nan)
        data_frame[column] = data_frame[column].astype(float)
        mean_value = data_frame[column].mean()
        data_frame[column].fillna(mean_value, inplace=True)
    return data_frame


def feature_selection(data_frame):
    """
    getting the dataframe and doing feature selection
    :param data_frame:
    :return data_frame
    """
    selected_features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                         'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    feature_dataframe = data_frame[selected_features]
    return feature_dataframe


def scaling(data_frame):
    """
    getting the dataframe and normalizing it
    @param data_frame
    @return data_frame
    """
    scale = StandardScaler()
    scaled_data = scale.fit_transform(data_frame)

    return scaled_data


def create_batches(dataset, batch_size):
    """
    creating batches to cover the complete dataset
    :param dataset
    :param batch_size
    :return batches
    """
    num_samples = len(dataset)
    num_batches = num_samples // batch_size

    # Create batches
    batches = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        batch = dataset[start_idx:end_idx]
        batches.append(batch)

    # Handle the remaining samples if the dataset size is not divisible by the batch size
    remaining_samples = num_samples % batch_size
    if remaining_samples > 0:
        batch = dataset[-remaining_samples:]
        batches.append(batch)

    return batches
