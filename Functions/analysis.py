'''
For visualisations
'''
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Can we use sns!????
def plot_correlation_heatmap(corr_matrix, top_features):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=['Feature {}'.format(i) for i in range(1, len(top_features)+1)], yticklabels=['Target Variable'])
    plt.title('Pearson Correlation Heatmap')
    plt.xlabel('Features')
    plt.ylabel('Target Variable')
    plt.show()


def fetch_confusion_matrix(classifier, test_set_x, test_set_y):
    y_pred = classifier.predict(test_set_x)
    ConfusionMatrixDisplay.from_predictions(test_set_y, y_pred)
    plt.title("Confusion Matrix")
    plt.show()