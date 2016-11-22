from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from scipy.spatial.distance import euclidean
from collections import defaultdict
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from itertools import combinations, izip






def load_data():
    df = pd.read_csv('more_than_five.csv')
    df = df.drop('Unnamed: 0', axis=1)
    return df


def kmeans(df, k=10):
    cols = [ '1w_open',
       '1w_crit', '1wacute', '1w_bg_entry', '1w_carbent', '1w_wtent','diabetes.type_Gestational',
       'diabetes.type_LADA', 'diabetes.type_Other',
       'diabetes.type_Pre-diabetic', 'diabetes.type_Prediabetes',
       'diabetes.type_Type I', 'diabetes.type_Type II', 'gender_Female',
       'gender_Male']
    df_for_kmeans = df.copy().loc[:,(cols)]
    loc = []
    for row in df_for_kmeans.index:
        new_row = df_for_kmeans.loc[row,:]
        if all(new_row.isnull()) == False and (new_row[0] > .1 or new_row[0] < -.1):
            # if new_row[0] > .1 or new_row[0] < -.1:
            loc.append(True)
        else:
            loc.append(False)
    df_for_kmeans2 = df_for_kmeans.copy()[loc]
    model3 = KMeans(n_clusters=10)
    df_for_kmeans2['cluster'] = model3.fit_predict(df_for_kmeans2)
    df['cluster'] = df_for_kmeans2['cluster'] + 1

    clusters= model3.cluster_centers_
    clusterz = list(clusters)
    d= defaultdict()
    for i, x in enumerate(clusterz):
        d[i] =x

    clusters = d

    return df, clusters



def filling_na(df):
    '''df = df.fillna(df['Label'].value_counts().index[0])
    to fill NaNs with the most frequent value from one column.'''
    df.fillna(-999, inplace=True)
    df.replace({'Na':np.nan}, inplace=True )
    return df

def convert_to_int(df,columns):
    columns.remove('dateOfFirstLaunch')
    for col in columns:
        df[col] = df[col].astype(int)
    return df

def load_data_train(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def Random_Forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=10, criterion='gini',
                                max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_features='auto',
                                max_leaf_nodes=None, bootstrap=True,
                                oob_score=False, n_jobs=-1, random_state=None,
                                verbose=0, warm_start=False, class_weight=None)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    rf_score = rf.score(X_test, y_test)
    print "score RF:", rf.score(X_test, y_test)
    #feat_importances = rf.feature_importances_
    return rf, rf_score, y_predict

def get_feature_importances(estimator,df):
    val = estimator.feature_importances_
    index = np.argsort(val)
    col_names = []
    for i in index:
        print df.columns[i], val[i]
        col_names.append(df.columns[i])
    return val, index, col_names


def plot_feature_importances(model,keep_cols):

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


from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, X_test, y_true):
    '''Code stolen brazenly from sklearn example.'''
    cm = confusion_matrix(y_true, model.predict(X_test))

    print(cm)

    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    df = load_data()

    columns = [u'user_id', u'appopen', u'max_open/day', u'use_age', u'Churn',
       u'1d_open', u'2dacute', u'2d_crit', u'2d_bg_entry', u'min_bg_2d',
       u'max_bg_2d', u'1d_carbent', u'1d_wtent', u'1w_open', u'1w_crit',
       u'1wacute', u'1w_bg_entry', u'min_bg_1w', u'max_bg_1w', u'1w_carbent',
       u'1w_wtent', u'tot_wt_ent', u'max_wtent/day', u'days_f0wt',
       u'max_carbent/day', u'days_f0crb', u'tot_carbent', u'min_bg', u'max_bg',
       u'avg_bg', u'tot_critical', u'tot_acute', u'tot_bg_open',
       u'max_bg_open', u'bg_user_age', u'BG_Schedule', u'dateOfFirstLaunch',
       u'Meal_Reminders_Set', u'TrackingExerciseMinutes',
       u'Total_BG_Reminders', u'carbBudget', u'diabetes.type_Gestational',
       u'diabetes.type_LADA', u'diabetes.type_Other',
       u'diabetes.type_Pre-diabetic', u'diabetes.type_Prediabetes',
       u'diabetes.type_Type I', u'diabetes.type_Type II', u'gender_Female',
       u'gender_Male', u'CarbEntryPreference_estimate',
       u'CarbEntryPreference_numeric', u'UserTreatmentType_Both',
       u'UserTreatmentType_Meds', u'engagemnt', u'1w_eng']

    columns_without_churn = [ u'1d_open', u'2dacute', u'2d_crit', u'2d_bg_entry', u'min_bg_2d',
       u'max_bg_2d', u'1d_carbent', u'1d_wtent', u'1w_open', u'1w_crit',
       u'1wacute', u'1w_bg_entry', u'min_bg_1w', u'max_bg_1w', u'1w_carbent',
       u'1w_wtent', u'min_bg', u'max_bg',
       u'avg_bg', u'tot_critical', u'tot_acute',  u'BG_Schedule',
       u'Meal_Reminders_Set', u'TrackingExerciseMinutes',
       u'Total_BG_Reminders', u'carbBudget', u'diabetes.type_Gestational',
       u'diabetes.type_LADA', u'diabetes.type_Other',
       u'diabetes.type_Pre-diabetic', u'diabetes.type_Prediabetes',
       u'diabetes.type_Type I', u'diabetes.type_Type II', u'gender_Female',
       u'gender_Male', u'CarbEntryPreference_estimate',
       u'CarbEntryPreference_numeric', u'UserTreatmentType_Both',
       u'UserTreatmentType_Meds', u'1w_eng']
    churn = df['Churn']


    df = filling_na(df)
    df = convert_to_int(df,columns)

    df['dateOfFirstLaunch'] = pd.to_datetime(df['dateOfFirstLaunch'])
    df['dateOfFirstLaunch'] = df['dateOfFirstLaunch']- (df['dateOfFirstLaunch'].min())
    df['dateOfFirstLaunch'] = (df['dateOfFirstLaunch'] / np.timedelta64(1, 'D')).astype(int)
    df['high_risk']= (df['tot_critical']>=1).astype(int)

    k_means_df = kmeans(df)



    X = df[columns_without_churn].values
    y = churn.values
    X_train, X_test, y_train, y_test = load_data_train(X,y)
    rf, rf_score, rf_pred = Random_Forest(X_train, y_train, X_test, y_test)
    feat_import = get_feature_importances(rf, df[columns_without_churn])



    '''TODO: RandomForestRegressor, linear regression on scalar variables, logistic regression on categorical , separate
    the categorical '''
