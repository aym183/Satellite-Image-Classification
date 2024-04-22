'''
For visualisations
'''
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc, roc_curve, det_curve
import numpy as np

def plot_single_correlation_heatmap(corr_matrix, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.xticks(np.arange(corr_matrix.shape[0]), np.arange(1, corr_matrix.shape[0] + 1), rotation=45)
    plt.yticks(np.arange(corr_matrix.shape[0]), np.arange(1, corr_matrix.shape[0] + 1))
    plt.title('Pearson Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.title(title)
    for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                plt.text(j, i, '{:.2f}'.format(corr_matrix[i, j]), ha='center', va='center', color='black')
    plt.show()

def plot_correlation_heatmap(ax, corr_matrix, title):
    im = ax.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    ax.set_xticks(np.arange(corr_matrix.shape[0]))
    ax.set_yticks(np.arange(corr_matrix.shape[0]))
    ax.set_xticklabels(np.arange(1, corr_matrix.shape[0] + 1), rotation=45)
    ax.set_yticklabels(np.arange(1, corr_matrix.shape[0] + 1))
    ax.set_title(title)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            ax.text(j, i, '{:.2f}'.format(corr_matrix[i, j]), ha='center', va='center', color='black')


def plot_confusion_matrix(classifier, test_set_x, test_set_y):
    confusion_matrix_values = []
    y_pred = classifier.predict(test_set_x)
    ConfusionMatrixDisplay.from_predictions(test_set_y, y_pred)
    plt.title("Confusion Matrix")
    plt.show()

    conf_mtrx = confusion_matrix(test_set_y, y_pred)
    metrics = []
    
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

    # table.auto_set_font_size(False)
    # table.set_fontsize(8)
    table.scale(1.2, 1)
    ax.axis('off')
    plt.show()
    
    # FP = conf_mtrx.sum(axis=0) - TP
    # FN = conf_mtrx.sum(axis=1) - TP
    # TP = np.diag(conf_mtrx)
    # TN = conf_mtrx.sum() - (FP + FN + TP)

    # print(f"True Positive -> {TP}")
    # print(f"True Negative -> {TN}")
    # print(f"False Positive -> {FP}")
    # print(f"False Negative -> {FN}")

def plot_precision_recall_curve(y_true, y_pred, ax):
    for idx in range(len(np.unique(y_true))):
        precision, recall, _ = precision_recall_curve((y_true == idx).astype(int), y_pred[:, idx])
        ax.plot(recall, precision, lw=2, label='Class {}'.format(idx))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='best')
    # plt.show()

def plot_roc_curve(y_true, y_pred, ax):
    for idx in range(len(np.unique(y_true))):
        fpr, tpr, _ = roc_curve((y_true == idx).astype(int), y_pred[:, idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label='Class {}'.format(idx, roc_auc)) # (AUC = {:.2f}

    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    # plt.show()

def plot_det_curve(y_true, y_pred, ax):
    for idx in range(len(np.unique(y_true))):
        fpr, fnr, _ = det_curve((y_true == idx).astype(int), y_pred[:, idx])
        ax.plot(fpr, fnr, lw=2, label='Class {}'.format(idx))

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('False Negative Rate')
    ax.set_title('DET Curve')
    ax.legend(loc='best')
    # plt.show()

def plot_subfigures(plots, plot_titles, y_pred, y_true, fig_size=(15,10)):
    columns = 2
    plots_length = len(plots)
    num_rows = (plots_length + columns - 1) // plots_length
    figure, axes = plt.subplots(num_rows, columns, figsize=fig_size) 

    for idx, plot_func in enumerate(plots):
        plt.sca = axes[idx]
        plot_func(y_true, y_pred)
        plt.title(plot_titles[idx])

    plt.tight_layout()
    plt.show()

def plot_predicted_vs_actual(y_test, y_pred):
    plt.figure(figsize=(6, 4))
    plt.plot(y_test[:100], 'o', label='Actual')
    plt.plot(y_pred[:100], 'x', label='Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()