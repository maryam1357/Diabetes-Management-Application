import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from bokeh.charts import Bar, output_file, show
import datetime as dt
import numpy as np


def read_files():
    open_by_user = pd.read_csv('../data/AppOpensByUser.csv').drop(['Unnamed: 0'],axis=1)
    bg_values = pd.read_csv('../data/BGValuesByUser.csv').drop(['Unnamed: 0'],axis=1)
    carb_ent = pd.read_csv('../data/CarbEntriesByUser.csv').drop(['Unnamed: 0'],axis=1)
    user_profiles = pd.read_csv('../data/UserProfiles.csv').drop(['Unnamed: 0'],axis=1)
    weights = pd.read_csv('../data/WeightEntriesByUser.csv').drop(['Unnamed: 0'],axis=1)
    user_profiles.columns = [ u'user_id', u'BG_Schedule', u'dateOfFirstLaunch', u'diabetes.type', u'Meal_Reminders_Set',
       u'TrackingExerciseMinutes', u'UserTreatmentType',
       u'carbBudget', u'gender',
       u'CarbEntryPreference']


    return open_by_user , bg_values, carb_ent, user_profiles, weights

def to_datetme(df, col):
    df[col] = pd.to_datetime(df[col])
    return df

def user_id_to_int(df):
    df = df.fillna(55555)
    df['user_id']= df['user_id'].astype(int)
    return df

def acute_crit(bg_df):
    bg_df['critical']=(bg_df['value'] < 50.0) | (bg_df['value'] > 350.0)
    bg_df['acute']=(bg_df['value'] <70) | (bg_df['value'] >250.0)
    (bg_df['critical']==True).sum() # 7071
    (bg_df['acute']==True).sum() # 7992
    bg_df[['acute','critical']]= bg_df[['acute','critical']].astype(int)
    return bg_df

def bg_outlier(bg_df):
    bg_df = bg_df[bg_df['value']<1000]
    bg_df = bg_df[bg_df['value']>20]
    bg_df['bg_appopen'] = 1
    return bg_df



def min_max_date(open_df):
    '''returns the df with min/max dates per user '''
    min_date = open_by_user.groupby(['user_id'], as_index=False).min()
    max_date = open_by_user.groupby(['user_id'], as_index=False).max()
    date_open = pd.merge(min_date,max_date,on='user_id')
    date_open.columns = ['user_id', 'min_date','min_open/day', 'max_date', 'max_open/day']
    date_open = date_open.drop('min_open/day', axis=1)
    return date_open

def group_date(df, date_open):
    grouped_df = pd.merge(df, date_open , on= 'user_id')
    return grouped_df

def app_user_age(df):
    df['user_app_age']= df['max_date']- df['min_date']
    return df

def set_min_date_zero(df,label):
    name = 'days_f0{}'.format(label)
    df[name] = df['Date'] - df['min_date']
    df[name] = (df[name] / np.timedelta64(1, 'D')).astype(int)
    return df

def del_min_max_date(df):
    df = df.drop(['min_date','max_date'], axis=1)
    return df

def rename_col(df, col, name):
    df.rename(columns = {col:name}, inplace=True)
    return df

def group_by_userid(df):
    grouped_df = df.groupby('user_id').sum().reset_index()
    return grouped_df

def bin_bg(df):
    bins = [ 0,  14,  28,  42,  56,  70,  84,  98, 112]
    df['bins'] = pd.cut(df['days_f0bg'], bins )
    df_avg = bg_values.groupby(['user_id', 'bins'])['bg'].mean()
    df_avg = df_avg.to_frame().reset_index()
    bg_diff =  df_avg['bg'].diff().to_frame()
    bg_diff.rename(columns = {'bg':'bg_2wdiff'}, inplace=True)
    df_avg = pd.concat([df_avg, bg_diff], axis=1)
    df_avg['>20']= df_avg['bg_2wdiff'] >=20
    df_avg['<-20'] = df_avg['bg_2wdiff']<= -20
    df_avg = df_avg.groupby('user_id').sum().reset_index()
    bg_diff = df_avg[['>20', '<-20','user_id']]

    return bg_diff

def first_week(open_by_user, weights, carb_ent, bg_values):
    open_1w = open_by_user[open_by_user['days_f0opn']<=7]
    open_1w = open_1w.groupby('user_id').sum().reset_index()
    open_1w = open_1w.rename(columns={'appopen':'1w_open'})
    open_1w = open_1w.drop(['max_open/day', 'days_f0opn','churn'], axis=1)

    bg_1w = bg_values[bg_values['days_f0bg']<=7]
    bg_1w_sum = bg_1w.groupby('user_id').sum().reset_index()[['user_id','critical','acute','bg_appopen']]
    bg_max_7d =bg_1w.groupby('user_id').max().reset_index()[['bg', 'user_id']]
    bg_min_7d =bg_1w.groupby('user_id').min().reset_index()[['bg', 'user_id']]
    bg_max_min = pd.merge(bg_min_7d, bg_max_7d, how='left' , left_on='user_id', right_on='user_id')

    bg_1w = pd.merge(bg_1w_sum, bg_max_min,how='left' , left_on='user_id', right_on='user_id' )

    bg_1w = bg_1w.rename(columns={'bg_appopen':'1w_bg_entry', 'critical':'1w_crit', 'acute':'1wacute', 'bg_x':'min_bg_1w', 'bg_y':'max_bg_1w' })




    carb_1w = carb_ent[carb_ent['days_f0crb']<=7]
    carb_1w = carb_1w.groupby('user_id').sum().reset_index()
    carb_1w = carb_1w.rename(columns={'carbent':'1w_carbent'})
    carb_1w = carb_1w.drop(['max_open/day', 'days_f0crb'], axis=1)

    wt_1w = weights[weights['days_f0wt']<=7]
    wt_1w = wt_1w.groupby('user_id').sum().reset_index()
    wt_1w = wt_1w.rename(columns={'wt_ent':'1w_wtent'})
    wt_1w = wt_1w.drop(['max_open/day', 'days_f0wt'], axis=1)

    open_bg = pd.merge(open_1w, bg_1w , how='left', left_on='user_id', right_on='user_id')
    carb_wt = pd.merge(carb_1w, wt_1w, how='left', left_on='user_id', right_on='user_id')
    week_1_data = pd.merge(open_bg , carb_wt, how='left', left_on='user_id', right_on='user_id')

    return week_1_data

def one_month_from_start(open_by_user, weights, carb_ent, bg_values):

    open_1m = open_by_user[(open_by_user['days_f0opn']>30) &(open_by_user['days_f0opn']<=37) ]
    open_1m = open_1m.groupby('user_id').sum().reset_index()
    open_1m = open_1m.rename(columns={'appopen':'1month/week_open'})
    open_1m = open_1m.drop(['max_open/day', 'days_f0opn','churn'], axis=1)

    bg_1m = bg_values[(bg_values['days_f0bg']>30) &(bg_values['days_f0bg']<=37)]
    bg_1m_sum = bg_1m.groupby('user_id').sum().reset_index()[['user_id','critical','acute','bg_appopen']]
    bg_max_1m =bg_1m.groupby('user_id').max().reset_index()[['bg', 'user_id']]
    bg_min_1m =bg_1m.groupby('user_id').min().reset_index()[['bg', 'user_id']]
    bg_max_min = pd.merge(bg_min_1m, bg_max_1m, how='left' , left_on='user_id', right_on='user_id')

    bg_1m = pd.merge(bg_1m_sum, bg_max_min,how='left' , left_on='user_id', right_on='user_id' )

    bg_1m = bg_1m.rename(columns={'bg_appopen':'1m_bg_entry', 'critical':'1m_crit', 'acute':'1macute', 'bg_x':'min_bg_1m', 'bg_y':'max_bg_1m' })



    open_bg = pd.merge(open_1m, bg_1m , how='left', left_on='user_id', right_on='user_id')
    open_bg['1month_engagement']= open_bg['1month/week_open']*1.0/7

    month_1_data = open_bg

    return month_1_data

def three_month_from_start(open_by_user, weights, carb_ent, bg_values):

    open_3m = open_by_user[(open_by_user['days_f0opn']>90) &(open_by_user['days_f0opn']<=97) ]
    open_3m = open_1m.groupby('user_id').sum().reset_index()
    open_3m = open_1m.rename(columns={'appopen':'1month/week_open'})
    open_3m = open_1m.drop(['max_open/day', 'days_f0opn','churn'], axis=1)

    bg_3m = bg_values[(bg_values['days_f0bg']>90) &(bg_values['days_f0bg']<=97)]
    bg_3m_sum = bg_3m.groupby('user_id').sum().reset_index()[['user_id','critical','acute','bg_appopen']]
    bg_max_3m =bg_3m.groupby('user_id').max().reset_index()[['bg', 'user_id']]
    bg_min_3m =bg_3m.groupby('user_id').min().reset_index()[['bg', 'user_id']]
    bg_max_min = pd.merge(bg_min_3m, bg_max_3m, how='left' , left_on='user_id', right_on='user_id')

    bg_3m = pd.merge(bg_3m_sum, bg_max_min,how='left' , left_on='user_id', right_on='user_id' )

    bg_3m = bg_3m.rename(columns={'bg_appopen':'3m_bg_entry', 'critical':'3m_crit', 'acute':'3macute', 'bg_x':'min_bg_3m', 'bg_y':'max_bg_3m' })



    open_bg = pd.merge(open_1m, bg_1m , how='left', left_on='user_id', right_on='user_id')
    open_bg['1month_engagement']= open_bg['1month/week_open']*1.0/7

    month_3_data = open_bg

    return month_3_data

def first_day(open_by_user, weights, carb_ent, bg_values):
    open_1d = open_by_user[open_by_user['days_f0opn']<=2]
    open_1d = open_1d.groupby('user_id').sum().reset_index()
    open_1d = open_1d.rename(columns={'appopen':'1d_open'})
    open_1d= open_1d.drop(['max_open/day', 'days_f0opn','churn'], axis=1)

    bg_1d = bg_values[bg_values['days_f0bg']<=2]
    bg_1d = bg_1d.groupby('user_id').sum().reset_index()
    bg_max_1d =bg_1d.groupby('user_id').max().reset_index()[['bg', 'user_id']]
    bg_min_1d =bg_1d.groupby('user_id').min().reset_index()[['bg', 'user_id']]
    bg_max_min = pd.merge(bg_min_1d, bg_max_1d, how='left' , left_on='user_id', right_on='user_id')
    bg_1d = bg_1d[['user_id', 'acute','critical','bg_appopen'   ]]
    bg_1d = pd.merge(bg_1d, bg_max_min,how='left' , left_on='user_id', right_on='user_id' )
    bg_1d = bg_1d.rename(columns={'bg_appopen':'2d_bg_entry', 'critical':'2d_crit', 'acute':'2dacute', 'bg_x':'min_bg_2d', 'bg_y':'max_bg_2d' })

    carb_1d = carb_ent[carb_ent['days_f0crb']<=2]
    carb_1d = carb_1d.groupby('user_id').sum().reset_index()
    carb_1d = carb_1d.rename(columns={'carbent':'1d_carbent'})
    carb_1d = carb_1d.drop(['max_open/day', 'days_f0crb'], axis=1)

    wt_1d = weights[weights['days_f0wt']<=2]
    wt_1d= wt_1d.groupby('user_id').sum().reset_index()
    wt_1d = wt_1d.rename(columns={'wt_ent':'1d_wtent'})
    wt_1d = wt_1d.drop(['max_open/day', 'days_f0wt'], axis=1)

    open_bg = pd.merge(open_1d, bg_1d , how='left', left_on='user_id', right_on='user_id')
    carb_wt = pd.merge(carb_1d, wt_1d, how='left', left_on='user_id', right_on='user_id')
    day_1_data = pd.merge(open_bg , carb_wt, how='left', left_on='user_id', right_on='user_id')

    return day_1_data

def group_open(open_by_user):
    max_app_open = open_by_user.groupby('user_id')['appopen'].sum().to_frame().reset_index()
    grouped_open = open_by_user.groupby('user_id').max().reset_index()
    grouped_open = grouped_open.drop(['appopen', 'Date', 'min_date', 'max_date'], axis=1)
    grouped_open = grouped_open.rename(columns={'days_f0opn':'use_age'})
    grouped_open = pd.merge(max_app_open, grouped_open, how='left', left_on='user_id', right_on='user_id')

    return grouped_open


def max_per_user_carb_wt(carb_ent, weights) :
    tot_carb_ent = carb_ent.groupby('user_id')['carbent'].sum().reset_index()
    carb_group = carb_ent.groupby('user_id').max().reset_index().drop(['Date', 'max_open/day'],axis = 1)
    tot_carb_ent = tot_carb_ent.rename(columns={'carbent':'tot_carbent'})
    carb_group = pd.merge(carb_group , tot_carb_ent , how='left', left_on='user_id', right_on='user_id' )
    carb_group = carb_group[carb_group['days_f0crb']>-10]
    carb_group = carb_group.rename(columns = {'carbent':'max_carbent/day'})

    tot_wt_ent = weights.groupby('user_id')['wt_ent'].sum().reset_index()
    wt_group = weights.groupby('user_id').max().reset_index().drop(['Date','max_open/day'],axis = 1)
    tot_wt_ent = tot_wt_ent.rename(columns={'wt_ent':'tot_wt_ent'})
    wt_ent_group = pd.merge(tot_wt_ent,wt_group , how='left', left_on='user_id', right_on='user_id' )
    wt_ent_group = wt_ent_group[wt_ent_group['days_f0wt']>-10]
    wt_ent_group = wt_ent_group.rename(columns= {'wt_ent':'max_wtent/day'})

    return carb_group, wt_ent_group




def merge(df1, df2):
    df_merged = pd.merge(df1,df2 , how='left', left_on='user_id', right_on='user_id')
    return df_merged

def min_max_avg(bg_values):
    avg_bg_user = bg_values.groupby('user_id').mean()['bg'].reset_index()
    max_bg =  bg_values.groupby('user_id').max()['bg'].reset_index()
    min_bg =  bg_values.groupby('user_id').min()['bg'].reset_index()
    bg_vals = pd.merge(min_bg, max_bg ,how='left', left_on='user_id', right_on='user_id' )
    bg_vals = pd.merge(bg_vals, avg_bg_user, how='left', left_on='user_id', right_on='user_id')
    return bg_vals

def bg_vals_for_group(bg_values):
    max_app_open_for_bg = bg_values.groupby('user_id').sum().reset_index()
    max_app_open_for_bg = max_app_open_for_bg.drop(['bg', 'max_open/day', 'days_f0bg'], axis=1)
    max_app_open_for_bg = max_app_open_for_bg.rename(columns={'bg_appopen':'tot_bg_open', 'critical':'tot_critical', 'acute':'tot_acute'})
    max_bg2 = bg_values.groupby(['user_id']).max().reset_index()
    max_bg2 = max_bg2.drop(['critical','acute', 'bg_appopen', 'bins', 'bg','Date'], axis=1)
    max_bg2 = max_bg2.rename(columns={'days_f0bg':'bg_user_age'})
    bg_join = pd.merge(max_app_open_for_bg , max_bg2, how='left', left_on='user_id', right_on='user_id')
    return bg_join


def fixing_user_profiles(user_profiles):
    bg_schedule_replace ={'3day':3, '1day':1, '2day':2, '6day':6, '5day':5, '12day':12, '4day':4,
       '7day':7, '2month':60, '8day':8, '6week':42, '10day':10, '2week':14, '8month':240,
       '3week':21, '4week':28, '1week':7, '7week':49, '10week':70, '6month':42, '5week':35,
       '1month':30, '11month':330, '9day':9, '11day':11, '12month':12, '4month':4, '3month':3,
       '12week':90}
    user_profiles['BG_Schedule'] = user_profiles['BG_Schedule'].replace(bg_schedule_replace)
    user_profiles = user_profiles.replace({'none':np.nan, 'None':np.nan ,'nan':np.nan, 'Lada':'LADA', 'NaT':np.nan})
    return user_profiles

def append_dummies(df,col):
    dummies = pd.get_dummies(df[col],prefix = col)
    df = pd.concat([df,dummies],axis=1)
    df.pop(col)
    return df

def get_session_interval(open_by_user,grouped_open):
    test = open_by_user
    test['count'] = 1
    test =test.groupby('user_id').sum().reset_index()
    test2 = pd.merge(test, grouped_open, how='left', left_on='user_id', right_on='user_id')
    test2['session_interval']= test2['use_age']*1.0/test2['count']
    session_interval = test2[['user_id', 'session_interval']]
    return session_interval

def churn_rate(open_by_user):
    month0 = open_by_user.min_date.min()
    month1 = month0+ dt.timedelta(days=30)
    month2 = month1+ dt.timedelta(days=30)
    month3= month2+ dt.timedelta(days=30)
    month4= month2+ dt.timedelta(days=30)
    month5= month2+ dt.timedelta(days=30)
    month1_users = open_by_user[(open_by_user.min_date>= month0) & (open_by_user.min_date<= month1)]
    month2_users = open_by_user[(open_by_user.min_date>= month1) & (open_by_user.min_date<= month2)]
    month3_users = open_by_user[(open_by_user.min_date>= month2) & (open_by_user.min_date<= month3)]
    month4_users = open_by_user[(open_by_user.min_date>= month3)]
    month1_users = month1_users.groupby('user_id').sum().reset_index()
    month2_users = month2_users.groupby('user_id').sum().reset_index()
    month3_users = month3_users.groupby('user_id').sum().reset_index()
    month4_users = month4_users.groupby('user_id').sum().reset_index()
    ch_rate_1 = month1_users.churn_f.sum()*1.0/month1_users.shape[0]
    ch_rate_2 = month2_users.churn_f.sum()*1.0/month2_users.shape[0]
    ch_rate_3 = month3_users.churn_f.sum()*1.0/month3_users.shape[0]
    ch_rate_4 = month4_users.churn_f.sum()*1.0/month4_users.shape[0]
    return ch_rate_1, ch_rate_2, ch_rate_3, ch_rate_4

def churn_rates(open_by_user):
    ch_rates = []
    day0 = open_by_user.min_date.min()
    for x in range(180):

        day0= day0+ dt.timedelta(days=x)
        day1= day0+ dt.timedelta(days=7)
        users_subdf = open_by_user[(open_by_user.min_date>=day0)&(open_by_user.min_date<=day1)]
        #print users_subdf
        day1_users = users_subdf.groupby('user_id').sum().reset_index()
        #print day1_users
        ch_rates.append((day1_users.churn.sum()*1.0)/(1+day1_users.shape[0]))

    return ch_rates


if __name__ == '__main__':
    open_by_user , bg_values, carb_ent, user_profiles, weights = read_files()

    bg_values = to_datetme(bg_values, 'Date')
    bg_values = user_id_to_int(bg_values)
    bg_values = acute_crit(bg_values)
    bg_values = bg_outlier(bg_values)
    date_open = min_max_date(open_by_user)
    date_open = to_datetme(date_open , 'min_date')
    date_open = to_datetme(date_open , 'max_date')


    bg_values = group_date(bg_values, date_open)
    bg_values = set_min_date_zero(bg_values,'bg')
    bg_values = del_min_max_date(bg_values)
    bg_values = rename_col(bg_values, 'value', 'bg' )
    bg_diff = bin_bg(bg_values)
    bg_values = pd.merge(bg_values, bg_diff, how='left', right_on='user_id', left_on='user_id')

    open_by_user = to_datetme(open_by_user, 'Date')
    open_by_user = user_id_to_int(open_by_user)
    open_by_user = group_date(open_by_user, date_open)
    open_by_user = set_min_date_zero(open_by_user,'opn')
    open_by_user = rename_col(open_by_user, 'value', 'appopen')
    churn_day = open_by_user['max_date'].max() - dt.timedelta(days=10)
    open_by_user['churn']= (open_by_user['max_date']< churn_day).astype(int)
    # open_by_user = open_by_user.drop('churn', axis =1)
    # one_time_users = open_by_user[open_by_user['appopen']<=1]
    # open_by_user = open_by_user[open_by_user['appopen']>1]
    # open_by_user = open_by_user[open_by_user['appopen']<230]

    weights = user_id_to_int(weights)
    weights = to_datetme(weights, 'Date')
    weights = group_date(weights, date_open)
    weights = set_min_date_zero(weights,'wt')
    weights = del_min_max_date(weights)
    weights = rename_col(weights,'value','wt_ent')




    carb_ent = user_id_to_int(carb_ent)
    carb_ent = to_datetme(carb_ent,'Date')
    carb_ent = group_date(carb_ent, date_open)
    carb_ent = set_min_date_zero(carb_ent,'crb')
    carb_ent = del_min_max_date(carb_ent)
    carb_ent = rename_col(carb_ent, 'value', 'carbent')

    week_1_data = first_week(open_by_user, weights, carb_ent, bg_values)
    day_1_data  =first_day(open_by_user, weights, carb_ent, bg_values)
    week_1_data = merge(day_1_data, week_1_data)
    grouped_open = group_open(open_by_user)
    grouped_open['Churn'] = (grouped_open.churn>0).astype(int)
    grouped_open = grouped_open.drop(['churn'], axis=1)


    grouped_add_1week = merge(grouped_open, week_1_data)
    grouped_add_1week = grouped_add_1week.set_index('user_id')

    session_interval = get_session_interval(open_by_user,grouped_open)
    grouped_add_1week = grouped_add_1week.reset_index()
    grouped_add_1week = merge(grouped_add_1week, session_interval)
    grouped_add_1week = grouped_add_1week.reset_index()
    month_1_data = one_month_from_start(open_by_user, weights, carb_ent, bg_values)
    grouped_add_1week = merge(grouped_add_1week, month_1_data)
    month_3_data = three_month_from_start(open_by_user, weights, carb_ent, bg_values)



    #add carbs/weight grouped
    carb_group, wt_ent_group = max_per_user_carb_wt(carb_ent, weights)
    wt_ent_test = wt_ent_group.set_index('user_id')
    carb_group_test = carb_group.set_index('user_id')
    group_c_w = pd.concat([wt_ent_test, carb_group_test], axis=1)
    bg_min_max = min_max_avg(bg_values)
    bg_min_max = bg_min_max.rename(columns={'bg_x':'min_bg', 'bg_y':'max_bg','bg':'avg_bg'})
    bg_join = bg_vals_for_group(bg_values)
    bg_to_join_final = merge(bg_min_max, bg_join)
    bg_to_join_final = bg_to_join_final.set_index('user_id')
    bg_to_join_final =  bg_to_join_final.rename(columns={'max_open/day':'max_bg_open'})

    user_profiles = fixing_user_profiles(user_profiles)
    user_profiles = to_datetme(user_profiles, 'dateOfFirstLaunch')

    user_profiles = append_dummies(user_profiles, 'diabetes.type')
    user_profiles = append_dummies(user_profiles, 'gender')
    user_profiles = append_dummies(user_profiles, 'CarbEntryPreference')
    user_profiles = append_dummies(user_profiles, 'UserTreatmentType')
    #user_profiles = user_profiles.drop('numAppOpenLifeTime', axis=1)
    user_profiles = user_profiles.set_index('user_id')

    # tables upto here are: group_c_w, grouped_add_1week , bg_to_join_final, user_profiles
    #grouping the first three together:

    group_all_tables = pd.concat([grouped_add_1week, group_c_w, bg_to_join_final,user_profiles], axis=1)
    group_all_tables = group_all_tables.reset_index()
    group_all_tables = group_all_tables.dropna(subset=['appopen'])
    group_all_tables['engagement'] = group_all_tables['appopen']*1.0/group_all_tables['use_age']
    group_all_tables['bg/appopen']= group_all_tables['tot_bg_open']*1.0/group_all_tables['appopen']


    one_time_users = group_all_tables[group_all_tables['appopen']<5]
    more_thn_five = group_all_tables[group_all_tables['appopen']>=5]
    more_thn_five['1w_engagement']= more_thn_five['1w_open']/7
    more_thn_five = more_thn_five.drop(['>20_y', '<-20_y'], axis=1)
