import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from src.utils.constant import model_location

dbscan_model = pickle.load(open(model_location, 'rb'))


def model_validation(data_frame):
    """
    getting the list from pre-processing function and plotting cluster image and calculating accuracy
    :param data_frame
    :return accuracy
    """
    batches = data_frame[0]
    data = data_frame[1]

    labels = dbscan_model.labels_

    silhouette_scores = []
    for idx, batch in enumerate(batches):
        if idx > 0:
            print(f"Batch {idx + 1}:")
            print(batch)
            new_model = dbscan_model.fit(batch)
            new_labels = new_model.labels_
            labels = np.concatenate((labels, new_labels))

            # Check if there is more than one unique label
            if len(np.unique(new_labels)) > 1:
                silhouette_scores.append(silhouette_score(batch, new_labels))
            else:
                print("Skipping silhouette score calculation for this batch due to only one unique label.")

            print()
            # model.append(new)
            print()

    average_silhouette_score = np.mean(silhouette_scores)

    outliers = data[labels == -1]
    cluster = data[labels == 0]

    fig, ax = plt.subplots()

    # Plot outliers
    cluster.plot.scatter(x='P1', y='P2', color='red', label='cluster', ax=ax)
    outliers.plot.scatter(x='P1', y='P2', color='blue', label='Outliers', ax=ax)

    # Set labels and title
    plt.xlabel('P1')
    plt.ylabel('P2')
    plt.title('Scatter Plot of DataFrame')

    # Set legend
    plt.legend()

    # Display the plot
    plt.show()

    return f"Average Silhouette Score: {average_silhouette_score}"
