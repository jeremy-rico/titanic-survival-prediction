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

    #####################
    # FEATURE FILTERING #
    #####################

    # A note on metrics:
    # Lets imagine we are a time traveler and we are using this model to determine if someone should be able to board
    # the Titanic. If our model predicts they won't survive, we won't let them on the ship.
    #
    # False Positive: Someone who was going to survive the disaster was not allowed on the ship, they survive anyways but miss their trip
    # False Negative: We let someone on the boat who will not survive. They die becuase of our poor model
    #
    # The penalty for FN is obviously worse than FP
    # Therefore, we will optimize our model for Recall
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB

    from xgboost import XGBClassifier

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
            print(f"\n{alg_name}")
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
    
    log_model = modeling(LogisticRegression, 'Logistic Regression')
    # 83% f1 right of the bat not bad

    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import StratifiedKFold
    import matplotlib.pyplot as plt
    log = LogisticRegression()
    metric = "recall"
    print(f"\nOptimizing feature space for {metric}...")
    rfecv = RFECV(
        estimator=log,
        cv=StratifiedKFold(10, random_state=50, shuffle=True),
        scoring=metric
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

    X_rfe = X.iloc[:, rfecv.support_]
    dropped = [f for f in X.columns.to_list() if f not in X_rfe.columns.to_list()]
    print(f"Final training data shape: {X_rfe.shape}")
    print(f"Dropped the following features: {dropped}")

    #########################
    # ALGORITHM EXPERIMENTS #
    #########################
    log_model = modeling(LogisticRegression, 'Logistic Regression')
    
    svc_model = modeling(SVC, "Support Vector Classifier")

    rf_model = modeling(RandomForestClassifier, "Random Forest Classifier")

    dt_model = modeling(DecisionTreeClassifier, "Decision Tree Classifier")

    nb_model = modeling(GaussianNB, "Naive Bayes")

    gb_model = modeling(XGBClassifier,
                        "Gradient Boosted Tree",
                        params={
                            'n_estimators': 2,
                            'max_depth': 2,
                            'learning_rate': 1,
                            'objective':'binary:logistic',
                        })

    ##########################
    # HYPER PARAMETER TUNING #
    ##########################

    print("\nTuning Hyperparameters...")

    model = RandomForestClassifier()
    #print(model.get_params())
      
    from sklearn.model_selection import RepeatedStratifiedKFold
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats = 3, random_state=1)

    from scipy.stats import loguniform
    param_grid = {
        'bootstrap': [True, False],
        'n_estimators': list(range(100, 1800, 200)),
        'max_depth': list(range(10, 100, 10)),
        'min_samples_leaf': [1,2,4],
        'criterion': ['gini', 'entropy', 'log_loss'],
        #'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100],
    }

    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    import time

    print("\nPerforming random search")
    random_start = time.time()
    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=100,
        scoring=metric,
        n_jobs = -1,
        cv = cv,
        random_state=1
    )
    random_result = search.fit(X_rfe, y)
    random_params = random_result.best_params_
    print(random_params)
    print(f"Time elapsed: {time.time()-random_start:.3f} sec")
    
    print("\nPerforming exhaustive search")
    grid_start = time.time()
    search = GridSearchCV(
        model,
        param_grid,
        scoring=metric,
        n_jobs = -1,
        cv = cv,
    )
    grid_result = search.fit(X_rfe, y)
    grid_params = grid_result.best_params_
    print(grid_params)
    print(f"Time elapsed: {time.time()-grid_start:.3f} sec")

    final_model = modeling(
        DecisionTreeClassifier,
        'Final Decision Tree Model',
        params = grid_params
    )
    

if __name__ == "__main__":
    main()
