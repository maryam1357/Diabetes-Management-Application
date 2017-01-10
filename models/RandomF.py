from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
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
from sklearn.ensemble import GradientBoostingClassifier


def first_week_session(df, open_by_user):
    open_by_user_1w = open_by_user[open_by_user['days_f0opn']<=7]
    test =open_by_user_1w.groupby('user_id').sum().reset_index()
    test['count'] = 7.0/test['count']
    test['churn'] = (test['churn']>0).astype(int)
    test.rename(columns={'count':'1w_sess_int'}, inplace=True)
    test['1w_sess_int'] = test['1w_sess_int'].astype(int)
    test = test[['user_id', '1w_sess_int']]
    return test



def load_data():
    df = pd.read_csv('more_thn_five.csv')
    df = df.drop('Unnamed: 0', axis=1)
    return df


def kmeans(df, columns_without_churn, k=10):
    cols = columns_without_churn
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

    return df



def filling_na(df):
    '''df = df.fillna(df['Label'].value_counts().index[0])
    to fill NaNs with the most frequent value from one column.'''
    df.fillna(-999, inplace=True)
    df.replace({'Na':np.nan}, inplace=True )
    return df

def convert_to_int(df,columns):
    columns.remove('dateOfFirstLaunch')

    df = df.drop('BG_Schedule', axis =1)
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

def view_feature_importances(df, model):
    """
    Args:
        df (pandas dataframe): dataframe which has the original data
        model (sklearn model): this is the sklearn classification model that
        has already been fit (work with tree based models)
    Returns:
        nothing, this just prints the feature importances in descending order
    """
    columns = df.columns
    features = model.feature_importances_
    featimps = []
    for column, feature in zip(columns, features):
        featimps.append([column, feature])
    print(pd.DataFrame(featimps, columns=['Features',
                       'Importances']).sort_values(by='Importances',
                                                   ascending=False))


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

def gradient_boost(X_train, y_train, X_test, y_test):
    gb_grid  = {
    'learning_rate': [0.1, 0.05, 0.01, 0.005],
    'n_estimators': [500,750, 1000],
    'loss': ['deviance','exponential']}

    gb_grid = GridSearchCV(GradientBoostingClassifier(subsample=.4),
                             gb_grid,
                             n_jobs=-1,
                             verbose=True, cv= 5)



    gb_grid.fit(X_train, y_train)
    best_params = gb_grid.best_params_
    best_score = gb_grid.best_score_
    best_model = gb_grid.best_estimator_

    return best_model, best_score, best_params

def grid_search():
    rf_grid = {
    'max_depth': [4, 8, None],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [1, 2, 4],
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
                             verbose=True)
    rf_grid_cv.fit(X_train, y_train)

    best_params = rf_grid_cv.best_params_
    best_score = rf_grid_cv.best_score_
    best_model = rf_grid_cv.best_estimator_


    return best_model , best_params , best_score



if __name__ == '__main__':
    df = load_data()
    open_by_user = pd.read_csv('open_by_user.csv')

    columns = [ u'user_id', u'appopen',
       u'max_open/day', u'use_age', u'Churn', u'1d_open', u'2dacute',
       u'2d_crit', u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent',
       u'1d_wtent', u'1w_open', u'1w_crit', u'1wacute', u'1w_bg_entry',
       u'min_bg_1w', u'max_bg_1w', u'1w_carbent', u'1w_wtent',
       u'session_interval', u'1month/week_open', u'count_x', u'1m_crit',
       u'1macute', u'1m_bg_entry', u'min_bg_1m', u'max_bg_1m',
       u'1month_engagement', u'3month/week_open', u'count_y', u'3m_crit',
       u'3macute', u'3m_bg_entry', u'min_bg_3m', u'max_bg_3m',
       u'3month_engagement', u'tot_wt_ent', u'max_wtent/day', u'days_f0wt',
       u'max_carbent/day', u'days_f0crb', u'tot_carbent', u'min_bg', u'max_bg',
       u'avg_bg', u'tot_critical', u'tot_acute', u'tot_bg_open', u'>20_x',
       u'<-20_x', u'max_bg_open', u'bg_user_age',
       u'dateOfFirstLaunch', u'Meal_Reminders_Set', u'TrackingExerciseMinutes',
       u'carbBudget', u'diabetes.type_Gestational', u'diabetes.type_LADA',
       u'diabetes.type_Other', u'diabetes.type_Pre-diabetic',
       u'diabetes.type_Prediabetes', u'diabetes.type_Type I',
       u'diabetes.type_Type II', u'gender_Female', u'gender_Male',
       u'CarbEntryPreference_estimate', u'CarbEntryPreference_numeric',
       u'UserTreatmentType_Both', u'UserTreatmentType_Insulin',
       u'UserTreatmentType_Meds', u'engagement', u'bg/appopen',
       u'1w_engagement']

    columns_without_churn = [ u'1d_open', u'2dacute', u'2d_crit', u'2d_bg_entry', u'min_bg_2d',
       u'max_bg_2d', u'1d_carbent', u'1d_wtent', u'1w_open', u'1w_crit',
       u'1wacute', u'1w_bg_entry', u'min_bg_1w', u'max_bg_1w', u'1w_carbent',
       u'1w_wtent', u'min_bg', u'max_bg',
       u'avg_bg', u'tot_critical', u'tot_acute',
       u'Meal_Reminders_Set', u'TrackingExerciseMinutes', u'carbBudget', u'diabetes.type_Gestational',
       u'diabetes.type_LADA', u'diabetes.type_Other',
       u'diabetes.type_Pre-diabetic', u'diabetes.type_Prediabetes',
       u'diabetes.type_Type I', u'diabetes.type_Type II', u'gender_Female',
       u'gender_Male', u'CarbEntryPreference_estimate',
       u'CarbEntryPreference_numeric', u'UserTreatmentType_Both',
       u'UserTreatmentType_Meds', u'1w_engagement' ,'1w_sess_int']
    churn = df['Churn']
    first_week_session = first_week_session(df, open_by_user)
    df = pd.merge(df,first_week_session,how='left', left_on='user_id', right_on='user_id' )


    df = filling_na(df)
    df = convert_to_int(df,columns)

    df['dateOfFirstLaunch'] = pd.to_datetime(df['dateOfFirstLaunch'])
    df['dateOfFirstLaunch'] = df['dateOfFirstLaunch']- (df['dateOfFirstLaunch'].min())
    df['dateOfFirstLaunch'] = (df['dateOfFirstLaunch'] / np.timedelta64(1, 'D')).astype(int)
    df['high_risk']= (df['tot_critical']>=1).astype(int)

    km_df = kmeans(df, columns_without_churn)



    X = df[columns_without_churn].values
    y = churn.values
    X_train, X_test, y_train, y_test = load_data_train(X,y)
    rf, rf_score, rf_pred = Random_Forest(X_train, y_train, X_test, y_test)
    #view_feature_importances(df[columns_without_churn], rf)
    #plot_feature_importances(rf, columns_without_churn)
    #best_model, best_score, best_params = gradient_boost(X_train, y_train, X_test, y_test)

    # km_cols = columns_without_churn
    # km_cols.append('cluster')
    # X = km_df[km_cols].values
    # y = churn.values
    # X_train, X_test, y_train, y_test = load_data_train(X,y)
    rf, rf_score, rf_pred = Random_Forest(X_train, y_train, X_test, y_test)
    #best_model , best_params , best_score = grid_search()
