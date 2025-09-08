import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



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
