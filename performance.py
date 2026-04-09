import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def save_confusion_matrix(y_true, y_pred, labels, figure_title, png_title):
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    png_name = "results/" + png_title + ".png"

    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=labels, yticklabels=labels)
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(figure_title)
    
    plt.savefig(png_name)