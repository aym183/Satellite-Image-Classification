'''
This file contains all the functions required to plot the visualisations
'''
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc, roc_curve, det_curve
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
from typing import Union

def plot_feature_split_of_values(input_arrays: list[np.ndarray], labels: list[str]):
    ''' 
    Plots a histogram showing the split of values in each feature dataset 

    Parameters:
    input_arrays: list[np.ndarray]
        The datasets containing the the plottable inputs
    labels: list[str]
        The titles for the datasets
    '''
    fig, axs = plt.subplots(1, len(input_arrays), figsize=(15, 4)) 
    fig.suptitle('Split of Values', fontsize=16)
    
    for idx, input_array in enumerate(input_arrays):
        flattened_array = input_array.flatten()
        axs[idx].hist(flattened_array, bins=50, color='blue', alpha=0.7)
        axs[idx].set_title(f'{labels[idx]}')
        axs[idx].set_xlabel('Values')
        axs[idx].set_ylabel('Frequency')
        axs[idx].set_xlim(0, 3)
        axs[idx].set_xticks(np.arange(0, 3.1, 0.25))
    
    plt.tight_layout()
    plt.show()

def plot_class_split_of_values(input_arrays: list[np.ndarray], labels: list[str]):
    ''' 
    Plots a histogram showing the split of values of each class

    Parameters:
    input_arrays: list[np.ndarray]
        The datasets containing the the plottable inputs
    labels: list[str]
        The titles for the datasets
    '''
    fig, axs = plt.subplots(1, len(input_arrays), figsize=(15, 4)) 
    fig.suptitle('Split of Values', fontsize=16)
    
    for idx, input_array in enumerate(input_arrays):
        flattened_array = input_array.flatten()
        axs[idx].hist(flattened_array, bins=50, color='blue', alpha=0.7)
        axs[idx].set_title(f'{labels[idx]}')
        axs[idx].set_xlabel('Values')
        axs[idx].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
def plot_single_correlation_heatmap(corr_matrix: np.corrcoef, title: str):
    ''' 
    Plots a heatmap showing the corrlation between all features and target variables 

    Parameters:
    corr_matrix: np.corrcoef
        The correlation matrix between the features and the target variable
    title: str
        The title for the figure
    '''
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.xticks(np.arange(corr_matrix.shape[0]), np.arange(1, corr_matrix.shape[0] + 1), rotation=45)
    plt.yticks(np.arange(corr_matrix.shape[0]), np.arange(1, corr_matrix.shape[0] + 1))
    plt.title('Pearson Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.title(title)
    for idx in range(corr_matrix.shape[0]):
            for idj in range(corr_matrix.shape[1]):
                plt.text(idj, idx, '{:.2f}'.format(corr_matrix[idx, idj]), ha='center', va='center', color='black')
    plt.show()

def plot_correlation_heatmap(ax, corr_matrix, title):
    im = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Correlation')
    ax.set_xticks(np.arange(corr_matrix.shape[0]))
    ax.set_yticks(np.arange(corr_matrix.shape[0]))
    ax.set_xticklabels(np.arange(1, corr_matrix.shape[0] + 1), rotation=45)
    ax.set_yticklabels(np.arange(1, corr_matrix.shape[0] + 1))
    ax.set_title(title)
    for idx in range(corr_matrix.shape[0]):
        for idj in range(corr_matrix.shape[1]):
            ax.text(idj, idx, '{:.2f}'.format(corr_matrix[idx, idj]), ha='center', va='center', color='black')


def plot_confusion_matrix(classifier: Union[SVC, MLPClassifier], test_set_x: np.ndarray, test_set_y: np.ndarray, table_needed: bool):
    ''' 
    Plots a confusion matrix to show the key metrics (i.e. the True Positive, True Negative, False Positive, and False Negative) for each class

    Parameters:
    classifier: Union[SVC, MLPClassifier]
        The classifier that was trained during development - Only accepts SVC and MLP for now
    test_set_x: np.ndarray
        The features testing dataset
    test_set_y: np.ndarray
        The classes testing dataset
    table_needed: bool
        This determines whether a table should be created as well for each of the metrics mentioned above
    '''
    confusion_matrix_values = []
    y_pred = classifier.predict(test_set_x)
    ConfusionMatrixDisplay.from_predictions(test_set_y, y_pred)
    plt.title("Confusion Matrix")
    plt.show()

    conf_mtrx = confusion_matrix(test_set_y, y_pred)
    
    if table_needed:
        for idx in range(len(conf_mtrx)):
            TP = conf_mtrx[idx, idx]
            FP = np.sum(conf_mtrx[:, idx]) - TP
            FN = np.sum(conf_mtrx[idx, :]) - TP
            TN = np.sum(conf_mtrx) - (TP + FP + FN)
            confusion_matrix_values.append((TP, FP, FN, TN))

        fig, ax = plt.subplots()
        table = ax.table(cellText=confusion_matrix_values,
                        colLabels=['True Positive', 'False Positive', 'False Negative', 'True Negative'],
                        rowLabels=[
                            'Class 0', 'Class 1', 'Class 2', 'Class 3',
                            'Class 4', 'Class 5',
                            'Class 6', 'Class 7', 'Class 8',
                            'Class 9'
                        ],
                        loc='center')

        table.scale(1.2, 1)
        ax.axis('off')
        plt.show()

def plot_precision_recall_curve(test_set_y: np.ndarray, pred_set_y: np.ndarray, ax: plt.axes):
    ''' 
    Plots a precision recall curve to see the rate of (i) true positive predictions to the total positive predictions 
    and (ii) true positive predictions to the true positives.

    Parameters:
    test_set_y: np.ndarray
        The classes testing dataset
    pred_set_y: np.ndarray
        The predictions made on the testing data
    ax: plt.axes
        Axis when plotting subfigures
    '''
    for idx in range(len(np.unique(test_set_y))):
        precision, recall, _ = precision_recall_curve((test_set_y == idx).astype(int), pred_set_y[:, idx])
        ax.plot(recall, precision, lw=2, label='Class {}'.format(idx))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='best')

def plot_roc_curve(test_set_y, pred_set_y, ax):
    ''' 
    Plots an ROC curve to see the rate of true positives to false positives

    Parameters:
    test_set_y: np.ndarray
        The classes testing dataset
    pred_set_y: np.ndarray
        The predictions made on the testing data
    ax: plt.axes
        Axis when plotting subfigures
    '''
    for idx in range(len(np.unique(test_set_y))):
        fpr, tpr, _ = roc_curve((test_set_y == idx).astype(int), pred_set_y[:, idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label='Class {}'.format(idx, roc_auc))

    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')

def plot_det_curve(test_set_y, pred_set_y, ax):
    ''' 
    Plots a DET curve to see the rate of false negatives to false positives

    Parameters:
    test_set_y: np.ndarray
        The classes testing dataset
    pred_set_y: np.ndarray
        The predictions made on the testing data
    ax: plt.axes
        Axis when plotting subfigures
    '''
    for idx in range(len(np.unique(test_set_y))):
        fpr, fnr, _ = det_curve((test_set_y == idx).astype(int), pred_set_y[:, idx])
        ax.plot(fpr, fnr, lw=2, label='Class {}'.format(idx))

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('False Negative Rate')
    ax.set_title('DET Curve')
    ax.legend(loc='best')

def plot_predicted_vs_actual(test_set_y: np.ndarray, pred_set_y: np.ndarray, title: str, ax: plt.axes):
    ''' 
    Plots a scatter plot showing the predicted vs actual values. 
    It is done only with 100 occurences so that the outputs are visible.

    Parameters:
    test_set_y: np.ndarray
        The classes testing dataset
    pred_set_y: np.ndarray
        The predictions made on the testing data
    title: str
        Title of the plot
    ax: plt.axes
        Axis when plotting subfigures
    '''
    ax.plot(test_set_y[:100], 'o', label='Actual')
    ax.plot(pred_set_y[:100], 'x', label='Prediction')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.legend()

# ------ TASK 4 ------

# def plot_clustering_results(train_set_x, test_set_x, clusters):

#     kmeans = KMeans(n_clusters=clusters)
#     kmeans.fit(train_set_x)
#     cluster_labels = kmeans.predict(test_set_x)

#     plt.figure(figsize=(8, 6))
#     plt.scatter(train_set_x[:, 0], test_set_x[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.5)
#     plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red', label='Cluster Centers')
#     plt.title('PCA Reduced Testing Dataset with K-means Clustering')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.colorbar(label='Cluster')
#     plt.legend()
#     plt.show()