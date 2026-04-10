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

def plot_histogram(dataframe, classes, feature, xlim_bottom, xlim_top, ylim_top, bins=30, title="", task=""):
    fig, ax = plt.subplots(1, len(classes), figsize=(12, 5))
    ax[0].set_ylabel("Frequency")
    for i, genre in enumerate(classes):
        ax[i].hist(dataframe[dataframe['Genre'] == genre][feature], bins=bins, color='blue', alpha=0.7)
        ax[i].set_xlim(xlim_bottom, xlim_top)  # Set x-axis limit for better visualization
        ax[i].set_ylim(0, ylim_top)  # Set y-axis limit for better visualization
        ax[i].set_xlabel(f"{feature}")
        ax[i].set_title(genre)
        ax[i].grid(axis='y', alpha=0.75)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"results/{task}_{feature}_histogram.png")