import pandas as pd
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
import datetime as dt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as pr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from collections import defaultdict
import random
from scipy.spatial.distance import euclidean
from collections import defaultdict
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier


def read_files():
    df = pd.read_csv('more_thn_five.csv')
    df.replace({'Na':np.nan}, inplace=True )
    df = df.drop([u'Unnamed: 0','carbBudget','TrackingExerciseMinutes','Meal_Reminders_Set', u'Unnamed: 0.1', u'level_0','BG_Schedule','dateOfFirstLaunch','days_f0crb','max_carbent/day', u'index',u'count_x',u'count_y','3m_crit','days_f0wt', '3macute','max_wtent/day'], axis=1)
    df['is_active'] = (df.loc[:,'Churn']==0).astype(int)
    return df

def define_cols():
    cols = [u'user_id', u'appopen', u'max_open/day',
           u'use_age', u'Churn', u'1d_open', u'2dacute', u'2d_crit',
           u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent', u'1d_wtent',
           u'1w_open', u'1w_crit', u'1wacute', u'1w_bg_entry', u'min_bg_1w',
           u'max_bg_1w', u'1w_carbent', u'1w_wtent', u'session_interval',
           u'1month/week_open', u'1m_crit', u'1macute', u'1m_bg_entry',
           u'min_bg_1m', u'max_bg_1m', u'1month_engagement', u'3month/week_open', u'3macute', u'3m_bg_entry', u'min_bg_3m',
           u'max_bg_3m', u'3month_engagement', u'tot_wt_ent',
           u'min_bg', u'max_bg', u'avg_bg', u'tot_critical', u'tot_acute',
           u'tot_bg_open', u'max_bg_open',
           u'diabetes.type_Gestational', u'diabetes.type_LADA',
           u'diabetes.type_Other', u'diabetes.type_Pre-diabetic',
           u'diabetes.type_Prediabetes', u'diabetes.type_Type I',
           u'diabetes.type_Type II', u'gender_Female', u'gender_Male', u'engagement', u'bg/appopen']
    two_day_features = [u'user_id', u'max_open/day', u'1d_open', u'2dacute', u'2d_crit',
           u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent', u'1d_wtent']
    first_week = [u'user_id', u'max_open/day', u'1d_open', u'2dacute', u'2d_crit', u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent', u'1d_wtent',
           u'1w_open', u'1w_crit', u'1wacute', u'1w_bg_entry', u'min_bg_1w',
           u'max_bg_1w', u'1w_carbent', u'1w_wtent']
    X_cols = [ u'max_open/day', u'1d_open', u'2dacute', u'2d_crit',
           u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent', u'1d_wtent',
           u'1w_open', u'1w_crit', u'1wacute', u'1w_bg_entry', u'min_bg_1w',
           u'max_bg_1w', u'1w_carbent', u'1w_wtent', u'session_interval',
           u'min_bg', u'max_bg', u'avg_bg', u'tot_critical', u'tot_acute',
           u'diabetes.type_Gestational', u'diabetes.type_LADA',
           u'diabetes.type_Other', u'diabetes.type_Pre-diabetic',
           u'diabetes.type_Prediabetes', u'diabetes.type_Type I',
           u'diabetes.type_Type II', u'gender_Female', u'gender_Male']
    y_col = ['Churn']
    # Features that need to be imputed
    zero_impute = ['1month/week_open' ,'1m_crit','1macute', '1m_bg_entry', '1month_engagement','3month/week_open','3m_bg_entry',
         'tot_wt_ent','3month_engagement','tot_wt_ent', 'tot_carbent','tot_bg_open','>20_x','<-20_x', 'bg_user_age','bg/appopen',
       'tot_acute', '2d_bg_entry','tot_critical','2d_crit', '1w_wtent', '1w_carbent','2dacute', '1d_carbent', '1d_wtent' , '1w_crit', '1wacute', '1w_bg_entry' ]
    mean_impute = ['min_bg_1w','min_bg_3m', 'max_bg_3m','max_bg_1w','min_bg', 'max_bg', 'avg_bg', '2dacute', 'min_bg_2d' ,'max_bg_2d','min_bg_1m', 'max_bg_1m', 'max_bg_open']

    return cols, two_day_features, y_col, zero_impute, mean_impute, first_week , X_cols

def fill_na_mean(df, columns):
    '''fill NA values with mean of that column.
    Input: dataframe, list of columns
    Output: dataframe '''

    for col in columns:
        mean = df[col].mean()
        df[col].fillna(mean,inplace=True)
    return df

def fill_na_zero(df, columns):
    '''fill NA values with 0.
    Input: dataframe, list of columns
    Output: dataframe '''

    for col in columns:
        mean = df[col].mean()
        df[col].fillna(0,inplace=True)
    return df

def convert_to_int(df,columns):
    '''Converts the columns values into integers so it can be used in Random_Forest'''
    for col in columns:
        df[col] = df[col].astype(int)
    return df

def smote(X, y, target, k=None):
    """
    INPUT:
    X, y - your data
    target - the percentage of negative class
             observations in the output
    k - k in k nearest neighbors
    OUTPUT:
    X_oversampled, y_oversampled - oversampled data
    `smote` generates new observations from the negative (minority) class:
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf
    """
    if target >= sum(y)/float(len(y)):
        return X, y
    if k is None:
        k = len(X)**.5
    # fit kNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X[y==0], y[y==0])
    neighbors = knn.kneighbors()[1]
    negative_observations = X[y==0]
    # determine how many new negative observations to generate
    positive_count = sum(y)
    negative_count = len(y) - positive_count
    target_negative_count = target*positive_count / (1. - target)
    target_negative_count = int(round(target_negative_count))
    number_of_new_observations = target_negative_count - negative_count


    # generate synthetic observations
    synthetic_observations = np.empty((0, X.shape[1]))
    while len(synthetic_observations) < number_of_new_observations:
        obs_index = np.random.randint(len(negative_observations))
        observation = negative_observations.iloc[obs_index]
        neighbor_index = np.random.choice(neighbors[obs_index])
        neighbor = X.iloc[neighbor_index]
        obs_weights = np.random.random(len(neighbor))
        neighbor_weights = 1 - obs_weights
        new_observation = obs_weights*observation + neighbor_weights*neighbor
        synthetic_observations = np.vstack((synthetic_observations, new_observation))

    X_smoted = np.vstack((X, synthetic_observations))
    y_smoted = np.concatenate((y, [0]*len(synthetic_observations)))

    return X_smoted, y_smoted

def load_data_train(X,y):
    '''split test and train data'''

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def Random_Forest(X_train, y_train, X_test, y_test):
    '''RF classifier results of grid_search: {'bootstrap': True,
     'max_depth': None,
     'max_features': 'sqrt',
     'min_samples_leaf': 4,
     'min_samples_split': 1,
     'n_estimators': 400,
     'n_jobs': -1,
     'oob_score': True,
     'random_state': 67}'''

    rf = RandomForestClassifier(n_estimators=400, criterion='gini',
                                max_depth=None, min_samples_leaf=4,
                                min_weight_fraction_leaf=0.0, max_features='sqrt',
                                max_leaf_nodes=None, bootstrap=True,
                                oob_score=True, n_jobs=-1, random_state=67,
                                verbose=0, warm_start=False, class_weight=None)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    print "score RF:", rf.score(X_test, y_test)
    probs = rf.predict_proba(X_test)[:,1].reshape(5729,1)
    y_t =y_test.reshape(5729,1)
    probs = np.concatenate((probs, y_t), axis =1)
    prob_churn = pd.DataFrame(probs)
    prob_churn = prob_churn.rename(columns={0:'Probability of Churn', 1:'Churn'})
    sample = prob_churn.sample(10)


    #feat_importances = rf.feature_importances_
    return rf, rf_score, y_predict

def get_feature_importances(estimator,df):
    '''input: model, datafram , output: column number, importance, column name'''

    val = estimator.feature_importances_
    index = np.argsort(val)
    col_names = []
    for i in index:
        print df.columns[i], val[i]
        col_names.append(df.columns[i])
    return val, index, col_names

def plot_feature_importances(model,keep_cols):
    '''Bar plot of important features'''

    keep_cols = np.array((keep_cols))
    feat_import = model.feature_importances_
    top10_nx = np.argsort(feat_import)[::-1][0:10]
    feat_import = feat_import[top10_nx]
    #normalize:
    feat_import = feat_import /feat_import.max()
    colnames = keep_cols[top10_nx]
    x_ind = np.arange(10)

    plt.barh(x_ind, feat_import, height=.3, align='center')
    plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
    plt.yticks(x_ind, colnames[x_ind])
    return top10_nx

def plot_confusion_matrix(model, X_test, y_true):
    '''Code stolen brazenly from sklearn example.'''
    cm = confusion_matrix(y_true, model.predict(X_test))

    print(cm)
    sns.heatmap(cm)
    # Show confusion matrix in a separate window
    #The code below produces black and white confusion matrix
    # plt.matshow(cm)
    # plt.title('Confusion matrix')
    # plt.colorbar()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

def rf_grid_search():
    '''grid search for gradient_boost classifier, the results are already implemented'''

    rf_grid = {
    'max_depth': [4, 8, None],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [ 2, 4],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True], # Mandatory with oob_score=True
    'n_estimators': [50, 100, 200, 400],
    'random_state': [67],
    'oob_score': [True],
    'n_jobs': [-1]
    }
    rf_grid_cv = GridSearchCV(RandomForestClassifier(),
                             rf_grid,
                             n_jobs=-1,
                             verbose=True,
                             scoring='roc_auc')
    rf_grid_cv.fit(X_train, y_train)
    best_params = rf_grid_cv.best_params_
    best_score = rf_grid_cv.best_score_
    best_model = rf_grid_cv.best_estimator_

    return best_model, best_score, best_params

def gradient_boost(X_train, y_train, X_test, y_test):
    '''gradient boosting classifier '''

    gb_grid = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.05, loss='deviance', max_depth=6,
              max_features=None, max_leaf_nodes=None,
              min_impurity_split=1e-07, min_samples_leaf=1,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              n_estimators=500, presort='auto', random_state=None,
              subsample=0.3, verbose=0, warm_start=False)




    gb_grid.fit(X_train, y_train)
    score_gb = gb_grid.score(X_test, y_test)

    return gb_grid , score_gb

def logistic_reg(X_train, X_test, y_train, y_test):
    '''logistic regression classifier'''

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # so that your scaler model does not ever see or fit to the x_test data
    # and this prevents train:test leakage
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)
    score = lr.score(X_test_scaled, y_test)
    lr_pred = lr.predict(X_test)

    return lr

def adaboost():
    '''adaboost classifier resulted accuracy: 0.81'''
    ab = AdaBoostClassifier(n_estimators=100)
    ab.fit(X_train, y_train)
    score = ab.score(X_test, y_test)
    return ab, score

def ensemble():
    '''voting ensemble (soft vote) of 3 models'''
    model1 = AdaBoostClassifier()
    estimators.append(('Adaboost', model1))
    model2 = RandomForestClassifier()
    estimators.append(('RandomForest', model2))
    model3 = GradientBoostingClassifier()
    estimators.append(('GradientBoosting', model2))
    # create the ensemble model
    ensemble = VotingClassifier(estimators)
    results = cross_validation.cross_val_score(ensemble, X_test, y_test, cv=10)

    return results.mean()

def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list
    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''

    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []

    num_positive_cases = np.sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist()

def run_roc_curve(model,label):
    '''plots roc curve of given models.'''


    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    tpr, fpr, thresholds = roc_curve(probabilities, y_test)

    plt.plot(fpr, tpr, label=label)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of 3 different Models")
    plt.legend()




if __name__=='__main__':
    # Read file and some initial processing.
    df = read_files()

    # Select columns that are going to be used in the models
    cols, two_day_features, y_col, zero_impute, mean_impute, first_week , X_cols = define_cols()
    # Fill na values, and convert to int
    df = fill_na_zero(df, zero_impute)
    df = fill_na_mean(df, mean_impute)
    df = df.dropna()
    df = convert_to_int(df, df.columns)

    #define X and Y
    X = df[X_cols]
    y = df['Churn']

    #Smote and Models

    X_s, y_s = smote(X, y, .5)
    X_train, X_test, y_train, y_test = load_data_train(X_s,y_s)
    rf, rf_score, rf_pred = Random_Forest(X_train, y_train, X_test, y_test)

    feat_import = get_feature_importances(rf, df[X_cols])
    plot_confusion_matrix(rf, X_test, y_test)
    #best_model, best_score, best_params = rf_grid_search()
    gb_grid , score_gb  = gradient_boost(X_train, y_train, X_test, y_test)
    lr, score , lr_pred = logistic_reg(X_train, y_train, X_test, y_test)
    run_roc_curve(lr, 'LogisticRegression')
    run_roc_curve(rf, 'RandomForest')
    run_roc_curve(gb_grid, 'GradientBoosting')
