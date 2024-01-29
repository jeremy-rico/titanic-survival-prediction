import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

def main():

    ################
    # DATA LOADING #
    ################
    data_path = Path('./src/data/')
    fig_dir = data_path / 'figures'

    pd.set_option('display.max_columns', None)
    df_train = pd.read_csv(data_path / 'train.csv')

    #################
    # DATA CLEANING #
    #################

    # Remove rows where Age or Embarked is missing
    df_train = df_train[df_train['Age'].notna()]
    df_train = df_train[df_train['Embarked'].notna()]
    print(df_train['Age'].shape[0])

    # Remove columns we won't be using
    df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    print(df_train.info())
    print(df_train.shape)

    ######################
    # DATA PREPROCESSING #
    ######################
    
    # Encode categorical features
    df_train = pd.get_dummies(df_train, dtype=float)
    # NOTE:
    # Necessary to encode Pclass? Values are 1,2,3. Could scale to 0,1,2? OR one hot encode?
    # for initial experiments I won't touch it. But come back to it later and experiment with
    # different preprocessing techniques.
    
    # Combine male and female columns into a binary 'gender' column
    df_train = df_train.drop('Sex_male', axis=1).rename(columns={"Sex_female": "Gender"})
    
    # Scale numerical features
    scaler = MinMaxScaler() #experiment with different scalers later
    df_train['Age'] = scaler.fit_transform(df_train[['Age']])
    df_train['Fare'] = scaler.fit_transform(df_train[['Fare']])
    print(df_train.head())

    """
    # check corr again
    import seaborn as sns
    import matplotlib.pyplot as plt

    corr = df_train.corr()
    plt.figure(figsize=(16,9))
    sns.heatmap(corr)
    plt.savefig(fig_dir / 'corr_heatmap(2).jpg')
    """
    ######################
    # FEATURE IMPORTANCE #
    ######################

    # Use a generalized linear model to determine if a feature effects the target
    # varibale in a statistically significant way
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    glm_columns = [col for col in df_train.columns if col not in ['Survived']]
    glm_columns = ' + '.join(glm_columns)
    formula = f"Survived ~ {glm_columns}"
    glm_model = smf.glm(
        formula=formula,
        data=df_train,
        family=sm.families.Binomial()
    )
    res = glm_model.fit()
    print(res.summary())
    # P>|z| column value < 0.005 means that feature is impacting our target variable

    ex_coef = np.exp(res.params)
    print(ex_coef)
    # exponential coefficient values over one indicate increased survival rate
    # values less than one decreased survival rate

    # RESULTS:
    # Pclass: medium inverse relationship on survival (higher class (=lower class value) higher survival rate)
    # Age: strong inverse relationship on survival (lower age, higher survival rate)
    # SibSp: medium inverse relationship on survival (less siblings, higher survival rate)
    # Parch: insignificant relationship on survival
    # Fare: weak positive relationship on survival (higher fare, higher survival rate)
    # Gender: strong positive relationpship on survival (female, much higher survival rate)
    # Embarked: embarked out of C or S weak positive relationship, embarked from Q insignificant

    # NOTES:
    # Could possible remove Parch as it has little effect on survival

    #########################
    # ALGORITHM EXPERIMENTS #
    #########################

    # A note on metrics:
    # Lets imagine we are a time traveler and we are using this model to determine if someone should be able to board
    # the Titanic. If our model predicts they won't survive, we won't let them on the ship.
    #
    # False Positive: Someone who was going to survive the disaster was not allowed on the ship, they survive anyways but miss their trip
    # False Negative: We let someone on the boat who will not survive. They die becuase of our poor model
    #
    # Based on these penalities it is obviously worse to get many false negatives
    # Therefore, we will optimize our model for Recall
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    X = df_train.drop('Survived', axis=1)
    y = df_train['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

    def modeling(alg, alg_name, print_scores=True, params={}):
        model = alg(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        def print_scores(alg, y_true, y_pred):
            print(alg_name)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_true, y_pred)
            print(f"Accuracy: {acc:.2f}")
            print(f"Precision: {prec:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1: {f1:.2f}")
            print(f"ROC AUC: {roc_auc:.2f}")

        if print_scores:
            print_scores(alg, y_test, y_pred)
            
        return model
    
    log_model = modeling(LogisticRegression, 'Logistic Regression', print_scores=True)
    # 83% f1 right of the bat not bad

    #####################
    # FEATURE FILTERING #
    #####################
    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import StratifiedKFold
    import matplotlib.pyplot as plt
    log = LogisticRegression()
    rfecv = RFECV(
        estimator=log,
        cv=StratifiedKFold(10, random_state=50, shuffle=True),
        scoring="accuracy"
    )
    rfecv.fit(X,y)

    test_scores = rfecv.cv_results_["mean_test_score"]
    plt.figure(figsize=(8,6))
    plt.plot(
        range(1,len(test_scores)+1),
        test_scores
    )
    plt.grid()
    plt.xlabel("Number of Selected Features")
    plt.ylabel("CV Score")
    plt.title("Recursive Feature Elimination (RFE)")
    plt.savefig(fig_dir/"RFE.jpg")
    print(f"The optimal number of features: {rfecv.n_features_}")
    # 7 for accuracy
    # 5 for recall 
    exit()
    
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

    ####################
    # PLOTS AND CHARTS #
    ####################

    print("Generating charts...")
    #seperate numerical and categorical columns
    df_numerical = df_train[["Age", "Fare", "SibSp", "Parch"]]
    df_categorical = df_train[["Survived", "Pclass", "Sex", "Embarked"]]

    # Pie chart to examine distribution of target variable
    target = df_train["Survived"].value_counts().to_frame()
    target = target.reset_index()
    plt.pie(target['count'], labels=target['Survived'], autopct='%1.1f%%')
    plt.title('Target Distribution')
    plt.savefig(fig_dir / "pie_survived.jpg")
    plt.close()

    # Histograms of numerical data
    for col in df_numerical.columns:
        hist(df_numerical, col, fig_dir)

    # Bar plots of categorical data
    for col in df_categorical.columns:
        bar(df_categorical, col, fig_dir)

    
    # Scatter Plots
    sns.lmplot(x='Age', y='Fare', data=df_train,
               fit_reg=False, hue='Survived')
    plt.savefig(fig_dir / "scatter_ageXfare.jpg")
    plt.close()

    # Box Plots
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
