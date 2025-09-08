import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (accuracy_score ,
                            recall_score,
                            confusion_matrix, 
                            precision_score,
                            f1_score, 
                            accuracy_score,
                            classification_report,
                            roc_curve,
                            roc_auc_score,
                           auc)


def plot_confusion_matrix(y_test, y_pred):
    logreg_conf_matrix = confusion_matrix(y_test, y_pred) 
    plt.figure(figsize = (4,4)) 
    sns.set(font_scale=1.4) 
    ax = sns.heatmap(logreg_conf_matrix, cmap='Blues',annot=True,
                    fmt='d', square=True,
                    xticklabels=['loyal (0)', 'churn (1)'],
                        yticklabels=['loyal (0)', 'churn (1)']) 
    ax.set(xlabel='Predicted', ylabel='Actual') 
    ax.invert_yaxis() 
    ax.invert_xaxis()

def plot_roc_auc(y_test, y_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Compute AUC
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def show_feature_stats(feature, target, data, show_all= True):
    print(f"Distribution of {feature} is:")
    print(data[feature].value_counts(normalize = True))
    print()

    if show_all:

        print(f"Distribution of {feature} in regards to {target} is:")
        print(data.groupby([feature, target])[target].count())


def show_feature(feature, target , data):
    ct = pd.crosstab(data[feature], data[target])

    ax= ct.plot(kind = 'bar', stacked = True, figsize= (6,4),) #colormap='Set2')
    
    
    plt.title(f"{target} Distribution by {feature}")
    plt.xlabel(f"{feature} (0 = No, 1 = Yes)")
    plt.ylabel("Number of Customers")
    plt.legend(title="Churn")


    
    # Add percentages relative to each bar's total
    for idx, row in ct.iterrows():
        total = row.sum()
        cumulative = 0
        for val in row:
            percent = val / total
            ax.text(idx, cumulative + val/2, f"{percent:.0%}", 
                    ha='center', va='center', fontsize=10, color="black")
            cumulative += val

    plt.show()
