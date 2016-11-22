import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from bokeh.charts import Bar, output_file, show
import datetime as dt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as pr
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('more_thn_five.csv')


cols = [u'user_id', u'appopen', u'max_open/day',
       u'use_age', u'Churn', u'1d_open', u'2dacute', u'2d_crit',
       u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent', u'1d_wtent',
       u'1w_open', u'1w_crit', u'1wacute', u'1w_bg_entry', u'min_bg_1w',
       u'max_bg_1w', u'1w_carbent', u'1w_wtent', u'session_interval',
       u'1month/week_open', u'1m_crit', u'1macute', u'1m_bg_entry',
       u'min_bg_1m', u'max_bg_1m', u'1month_engagement', u'3month/week_open', u'3m_crit', u'3macute', u'3m_bg_entry', u'min_bg_3m',
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

X_cols = [ u'appopen', u'max_open/day', u'1d_open', u'2dacute', u'2d_crit',
       u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent', u'1d_wtent',
       u'1w_open', u'1w_crit', u'1wacute', u'1w_bg_entry', u'min_bg_1w',
       u'max_bg_1w', u'1w_carbent', u'1w_wtent', u'session_interval',
       u'min_bg', u'max_bg', u'avg_bg', u'tot_critical', u'tot_acute',
       u'diabetes.type_Gestational', u'diabetes.type_LADA',
       u'diabetes.type_Other', u'diabetes.type_Pre-diabetic',
       u'diabetes.type_Prediabetes', u'diabetes.type_Type I',
       u'diabetes.type_Type II', u'gender_Female', u'gender_Male']

y_col = ['Churn']

df = df[cols]

def fill_na_mean(df, columns):
    for col in columns:
        mean = df[col].mean()
        df[col].fillna(mean,inplace=True)
    return df

def fill_na_zero(df, columns):
    for col in columns:
        mean = df[col].mean()
        df[col].fillna(0,inplace=True)
    return df

def convert_to_int(df,columns):
    for col in columns:
        df[col] = df[col].astype(int)
    return df

def smote(X, y, target, k=None):
    """
    INPUT:
    X, y - your data
    target - the percentage of positive class
             observations in the output
    k - k in k nearest neighbors
    OUTPUT:
    X_oversampled, y_oversampled - oversampled data
    `smote` generates new observations from the positive (minority) class:
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf
    """
    if target <= sum(y)/float(len(y)):
        return X, y
    if k is None:
        k = len(X)**.5
    # fit kNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X[y==1], y[y==1])
    neighbors = knn.kneighbors()[0]
    positive_observations = X[y==1]
    # determine how many new positive observations to generate
    positive_count = sum(y)
    negative_count = len(y) - positive_count
    target_positive_count = target*negative_count / (1. - target)
    target_positive_count = int(round(target_positive_count))
    number_of_new_observations = target_positive_count - positive_count
    # generate synthetic observations
    synthetic_observations = np.empty((0, X.shape[1]))
    while len(synthetic_observations) < number_of_new_observations:
        obs_index = np.random.randint(len(positive_observations))
        observation = positive_observations[obs_index]
        neighbor_index = np.random.choice(neighbors[obs_index])
        neighbor = X[neighbor_index]
        obs_weights = np.random.random(len(neighbor))
        neighbor_weights = 1 - obs_weights
        new_observation = obs_weights*observation + neighbor_weights*neighbor
        synthetic_observations = np.vstack((synthetic_observations, new_observation))

    X_smoted = np.vstack((X, synthetic_observations))
    y_smoted = np.concatenate((y, [1]*len(synthetic_observations)))

    return X_smoted, y_smoted


if __name__=='__main__':

    df = pd.read_csv('more_thn_five.csv')
    df.replace({'Na':np.nan}, inplace=True )


    cols = [u'user_id', u'appopen', u'max_open/day',
           u'use_age', u'Churn', u'1d_open', u'2dacute', u'2d_crit',
           u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent', u'1d_wtent',
           u'1w_open', u'1w_crit', u'1wacute', u'1w_bg_entry', u'min_bg_1w',
           u'max_bg_1w', u'1w_carbent', u'1w_wtent', u'session_interval',
           u'1month/week_open', u'1m_crit', u'1macute', u'1m_bg_entry',
           u'min_bg_1m', u'max_bg_1m', u'1month_engagement', u'3month/week_open', u'3m_crit', u'3macute', u'3m_bg_entry', u'min_bg_3m',
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

    X_cols = [ u'appopen', u'max_open/day', u'1d_open', u'2dacute', u'2d_crit',
           u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent', u'1d_wtent',
           u'1w_open', u'1w_crit', u'1wacute', u'1w_bg_entry', u'min_bg_1w',
           u'max_bg_1w', u'1w_carbent', u'1w_wtent', u'session_interval',
           u'min_bg', u'max_bg', u'avg_bg', u'tot_critical', u'tot_acute',
           u'diabetes.type_Gestational', u'diabetes.type_LADA',
           u'diabetes.type_Other', u'diabetes.type_Pre-diabetic',
           u'diabetes.type_Prediabetes', u'diabetes.type_Type I',
           u'diabetes.type_Type II', u'gender_Female', u'gender_Male']
    zero_impute = ['tot_acute', '2d_bg_entry','tot_critical','2d_crit', '1w_wtent', '1w_carbent','2dacute', '1d_carbent', '1d_wtent' , '1w_crit', '1wacute', '1w_bg_entry']
    mean_impute = ['min_bg_1w','max_bg_1w','min_bg', 'max_bg', 'avg_bg', '2dacute', 'min_bg_2d' ,'max_bg_2d']
    df = fill_na_zero(df, zero_impute)
    df = fill_na_mean(df, mean_impute)
    y_col = ['Churn']
    df = convert_to_int(df, cols)
    X = df[X_cols]
    y= df[y_col]
