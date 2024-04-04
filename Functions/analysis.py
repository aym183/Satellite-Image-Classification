'''
For visualisations
'''
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, auc, roc_curve, det_curve
import numpy as np

# Can we use sns!????
def plot_correlation_heatmap(corr_matrix, top_features):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=['Feature {}'.format(i) for i in range(1, len(top_features)+1)], yticklabels=['Target Variable'])
    plt.title('Pearson Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Target Variable')
    plt.show()


def plot_confusion_matrix(classifier, test_set_x, test_set_y):
    y_pred = classifier.predict(test_set_x)
    ConfusionMatrixDisplay.from_predictions(test_set_y, y_pred)
    plt.title("Confusion Matrix")
    plt.show()

def plot_precision_recall_curve(y_true, y_pred):
    plt.figure(figsize=(6,4))
    for i in range(len(np.unique(y_true))):
        precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_pred[:, i])
        plt.plot(recall, precision, lw=2, label='Class {}'.format(i))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

def plot_roc_curve(y_true, y_pred):
    plt.figure(figsize=(6, 4))
    for i in range(len(np.unique(y_true))):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label='Class {} (AUC = {:.2f})'.format(i, roc_auc))

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()

def plot_det_curve(y_true, y_pred):
    plt.figure(figsize=(6, 4))
    for i in range(len(np.unique(y_true))):
        fpr, fnr, _ = det_curve((y_true == i).astype(int), y_pred[:, i])
        plt.plot(fpr, fnr, lw=2, label='Class {}'.format(i))

    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('DET Curve')
    plt.legend(loc='best')
    plt.show()
