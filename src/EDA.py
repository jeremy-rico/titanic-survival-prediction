import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from pathlib import Path

def main():
    data_path = Path('./data/')
    fig_dir = data_path / 'figures'

    pd.set_option('display.max_columns', None)
    df_train = pd.read_csv(data_path / 'train.csv')
    df_test = pd.read_csv(data_path / 'test.csv')

    print(df_train.head())
    print(df_train.columns)

    # Scatter Plot
    sns.lmplot(x='Age', y='Fare', data=df_train,
               fit_reg=False, hue='Survived')
    plt.savefig(fig_dir / "scatter_ageXfare.jpg")
    plt.close()

    # Box Plot
    df_stats = df_train.drop(['PassengerId','Survived','Pclass','Name','Sex','Ticket','Cabin','Embarked'], axis=1)
    sns.boxplot(data=df_stats)
    plt.xticks(rotation=-45) #tilt xlabels for readability
    plt.savefig(fig_dir / "boxplot.jpg")
    plt.close()

    # Histograms
    sns.histplot(df_train, x='Age', kde=True)
    plt.savefig(fig_dir / "hist_age.jpg")
    plt.close()

    sns.histplot(df_train, x='Fare', kde=True)
    plt.savefig(fig_dir / "hist_fare.jpg")
    plt.close()

    # Bar Plots
    sns.countplot(x='Survived', data=df_train)
    plt.savefig(fig_dir / "bar_survived.jpg")
    plt.close()

if __name__ == "__main__":
    main()
