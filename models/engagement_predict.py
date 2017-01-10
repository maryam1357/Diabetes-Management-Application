import pandas as pd
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
import datetime as dt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support as pr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation  import StratifiedKFold
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.grid_search import GridSearchCV
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
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

def read_files():
    open_by_user = pd.read_csv('open_by_user.csv')

    df = pd.read_csv('more_thn_five.csv')
    df.replace({'Na':np.nan}, inplace=True )
    df = df.drop([u'Unnamed: 0','carbBudget','TrackingExerciseMinutes','Meal_Reminders_Set', u'Unnamed: 0.1', u'level_0','BG_Schedule','dateOfFirstLaunch','days_f0crb','max_carbent/day', u'index',u'count_x',u'count_y','3m_crit','days_f0wt', '3macute','max_wtent/day'], axis=1)
    df['is_active'] = (df.loc[:,'Churn']==0).astype(int)

    return open_by_user, df


def first_week_session(df, open_by_user):
    open_by_user_1w = open_by_user[open_by_user['days_f0opn']<=7]
    test =open_by_user_1w.groupby('user_id').sum().reset_index()
    test['count'] = 7.0/test['count']
    test['churn'] = (test['churn']>0).astype(int)
    test.rename(columns={'count':'1w_sess_int'}, inplace=True)
    test['1w_sess_int'] = test['1w_sess_int'].astype(int)
    test = test[['user_id', '1w_sess_int']]
    return test




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


def load_data_train(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    return X_train, X_test, y_train, y_test

def Random_Forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=1000, criterion='gini',
                                max_depth=None, min_samples_split=3, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_features='sqrt',
                                max_leaf_nodes=None, bootstrap=True, n_jobs=-1, random_state=67,
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

def oversample(X_train, y_train):
    ros = RandomOverSampler()
    X_o, y_o = ros.fit_sample(X_train, y_train)
    return X_o , y_o

def undersample(X_train, y_train):
    rus = RandomUnderSampler()
    X_u, y_u = rus.fit_sample(X_train, y_train)
    return X_u , y_u



def rf_grid_search():
    rf_grid = {
    'max_depth': [10,20],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True], # Mandatory with oob_score=True
    'n_estimators': [200],
    'random_state': [67],
    'oob_score': [True],
    'n_jobs': [1]}

    rf_grid_cv = GridSearchCV(RandomForestClassifier(),
                             rf_grid,
                             n_jobs=-1,
                             verbose=True, cv= 5)



    rf_grid_cv.fit(X_train, y_train)
    best_params = rf_grid_cv.best_params_
    best_score = rf_grid_cv.best_score_
    best_model = rf_grid_cv.best_estimator_

    return best_model, best_score, best_params

def gradient_boost(X_train, y_train, X_test, y_test):
    gb_grid  = {
    'learning_rate': [0.1, 0.05,],
    'n_estimators': [100,200,300],
    'max_depth': [6, 8, 3],
    'subsample': [0.3],
    'loss': ['deviance']}

    gb_grid = GridSearchCV(GradientBoostingClassifier(subsample=.4),
                             gb_grid,
                             n_jobs=-1,
                             verbose=True, cv= 5)




    gb_grid.fit(X_train, y_train)
    best_params_gb = gb_grid.best_params_
    best_score_gb = gb_grid.best_score_
    best_model_gb = gb_grid.best_estimator_

    return best_model_gb , best_score_gb , best_params_gb

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

    two_day_features = [u'user_id', u'max_open/day', u'1d_open', u'2dacute', u'2d_crit', u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent', u'1d_wtent']

    first_week =[u'user_id', u'max_open/day', u'1d_open', u'2dacute', u'2d_crit', u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent', u'1d_wtent',
       u'1w_open', u'1w_crit', u'1wacute', u'1w_bg_entry', u'min_bg_1w',
       u'max_bg_1w', u'1w_carbent', u'1w_wtent', '1w_sess_int']

    X_cols = [ u'max_open/day', u'1d_open', u'2dacute', u'2d_crit',
       u'2d_bg_entry', u'min_bg_2d', u'max_bg_2d', u'1d_carbent', u'1d_wtent',
       u'1w_open', u'1w_crit', u'1wacute', u'1w_bg_entry', u'min_bg_1w',
       u'max_bg_1w', u'1w_carbent', u'1w_wtent',
       u'diabetes.type_Gestational', u'diabetes.type_LADA',
       u'diabetes.type_Other', u'diabetes.type_Pre-diabetic',
       u'diabetes.type_Prediabetes', u'diabetes.type_Type I',
       u'diabetes.type_Type II', u'gender_Female', u'gender_Male','1w_sess_int']

    zero_impute =['1month/week_open' ,'1m_crit','1macute', '1m_bg_entry', '1month_engagement','3month/week_open','3m_bg_entry',
     'tot_wt_ent','3month_engagement','tot_wt_ent', 'tot_carbent','tot_bg_open','>20_x','<-20_x', 'bg_user_age','bg/appopen',
   'tot_acute', '2d_bg_entry','tot_critical','2d_crit', '1w_wtent', '1w_carbent','2dacute', '1d_carbent', '1d_wtent' , '1w_crit', '1wacute', '1w_bg_entry' ]
    mean_impute = ['min_bg_1w','min_bg_3m', 'max_bg_3m','max_bg_1w','min_bg', 'max_bg', 'avg_bg', '2dacute', 'min_bg_2d', '1w_sess_int','max_bg_2d','min_bg_1m', 'max_bg_1m', 'max_bg_open']

    return cols, two_day_features, first_week, X_cols, zero_impute, mean_impute

def engagement_class(df):
    df['engagement_class']= 0
    df.loc[(df['1month/week_open']>0) & (df['1month/week_open']<6),'engagement_class']= 1
    df.loc[(df['1month/week_open']>=6) & (df['1month/week_open']<16),'engagement_class']= 2
    df.loc[(df['1month/week_open']>=16) ,'engagement_class']= 3
    df = df.dropna()
    return df





if __name__=='__main__':

    ##Read files , fill NA's define columns
    open_by_user, df = read_files()
    first_week_session = first_week_session(df, open_by_user)
    df = pd.merge(df,first_week_session,how='left', left_on='user_id', right_on='user_id' )
    cols, two_day_features, first_week, X_cols, zero_impute, mean_impute = define_cols()
    df = fill_na_zero(df, zero_impute)
    df = fill_na_mean(df, mean_impute)
    df = engagement_class(df)
    df = convert_to_int(df, df.columns)

    ##Define X and Y
    X = df[X_cols]
    y = df['engagement_class']


    ##Taking care of Imbalanced class (undersample works best here)

    #X_s, y_s = smote(X, y, .5)
    X_u, y_u = undersample(X, y)
    X_train, X_test, y_train, y_test = load_data_train(X_u,y_u)
    #X_o, y_o = oversample(X_train, y_train)

    ##Models:
    rf, rf_score, rf_pred = Random_Forest(X_train, y_train, X_test, y_test)
    #feat_import = get_feature_importances(rf, df[X_cols])
    #view_feature_importances(df[X_cols], rf)
    #plot_feature_importances(rf, X_cols)

    #best_model_gb , best_score_gb , best_params_gb = gradient_boost(X_train, y_train, X_test, y_test)

    #rf_grid_search()
