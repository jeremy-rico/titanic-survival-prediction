import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from pathlib import Path

def hist(df, col_name, out_dir):
    sns.histplot(df, x=col_name, kde=True)
    plt.savefig(out_dir / f"hist_{col_name}.jpg")
    plt.close()

def bar(df, col_name, out_dir):
    sns.countplot(x=col_name, data=df)
    plt.savefig(out_dir / f"bar_{col_name}.jpg")
    plt.close()
    
def main():

    ################
    # DATA LOADING #
    ################
    data_path = Path('./data/')
    fig_dir = data_path / 'figures'

    pd.set_option('display.max_columns', None)
    df_train = pd.read_csv(data_path / 'train.csv')

    ########################
    # Initial Observations #
    ########################
    
    print(df_train.head())
    print(df_train.columns)
    print(f"NUM SAMPLES: {df_train.shape[0]}")

    # NOTES:
    # Target column: Survived
    # Unique value columns: PassengerId, Name, Ticket, Cabin

    print(df_train["Ticket"].unique().shape[0]) #681 unique values, not good for a feature

    # Missing values
    for col in df_train.columns:
        not_na = df_train[df_train[col].notna()].shape[0]
        missing = df_train.shape[0] - not_na
        print(f"Column {col} missing {missing} values")

    # NOTES:
    # Age missing 177 values, will omit missing values
    # Cabin missing 687 values, will omit whole column

    #Class balance
    pos = df_train[df_train['Survived']==1].shape[0] / df_train.shape[0]
    neg = 1.0-pos
    print(f"Class balance: {pos*100:.1f}% survived, {neg*100:.1f}% deceased")
    
    ####################
    # PLOTS AND CHARTS #
    ####################

    #seperate numerical and categorical columns
    df_numerical = df_train[["Age", "Fare", "SibSp", "Parch"]]
    df_categorical = df_train[["Survived", "Pclass", "Sex", "Cabin", "Embarked"]]
    
    # Histograms
    for col in df_numerical.columns:
        hist(df_numerical, col, fig_dir)

    # Bar plots
    for col in df_categorical.columns:
        bar(df_categorical, col, fig_dir)

    # Scatter Plot
    sns.lmplot(x='Age', y='Fare', data=df_train,
               fit_reg=False, hue='Survived')
    plt.savefig(fig_dir / "scatter_ageXfare.jpg")
    plt.close()

    # Box Plot
    sns.boxplot(data=df_train[["Age","Fare"]])
    plt.savefig(fig_dir / "boxplot_age&fare.jpg")
    plt.close()

    #NOTES:
    # Outlier around 500 fare (guy got ripped off lol)
    
    sns.boxplot(data=df_train[["SibSp","Parch"]])
    #plt.xticks(rotation=-45) #tilt xlabels for readability
    plt.savefig(fig_dir / "boxplot_sib&par.jpg")
    plt.close()

    # NOTES:
    # These two are odd, small distribution with many outliers. Will experiment with to see how useful they are

    # Correlation Heatmap
    df_stats = df_train.drop(['PassengerId','Survived','Pclass','Name','Sex','Ticket','Cabin','Embarked'], axis=1)
    corr = df_stats.corr()
    sns.heatmap(corr)
    plt.savefig(fig_dir / 'corr_heatmap.jpg')
    plt.close()

    # NOTES:
    # No strong correlations found

    ############
    # FINDINGS #
    ############

    """
    Remove rows with missing Age values
    Remove columns with unique values: PassengerId, Name, Ticket
    Remove column with too many missing values: Cabin
    Watch for outliers in Fare and Parch columns
    Class imbalance: 38.4% survived, 61.6% deceased
    """
if __name__ == "__main__":
    main()
