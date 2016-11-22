
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from bokeh.charts import Bar, output_file, show
import datetime as dt
import numpy as np


def med_risk_users(more_than_10):
    g = more_than_10.copy()
    med_risk_patients = g[(g['tot_acute']> 0) & (g['tot_critical']==0)][['user_id','tot_acute','tot_critical','use_age','appopen']]
    med_users = med_risk_patients['user_id'].values.tolist()
    return med_risk_patients, med_users

def high_risk_users(more_than_10):
    g = more_than_10.copy()
    high_risk_patients = g[(g['tot_acute']> 1) & (g['tot_critical']>=1)][['user_id','tot_acute','tot_critical','use_age','appopen']]
    high_users = high_risk_patients['user_id'].values.tolist()
    return high_risk_patients, high_users

def low_risk_users(more_than_10):
    g = more_than_10.copy()
    low_risk_patients = g[(g['tot_acute']== 0) & (g['tot_critical']==0)][['user_id','tot_acute','tot_critical','use_age','appopen']]
    low_users = low_risk_patients['user_id'].values.tolist()
    return low_risk_patients, low_users

def sample_high(high_risk_patients):
    high_r = high_risk_patients[high_risk_patients['appopen']>150]
    high_r = high_r[high_r['use_age']>50]
    high_sample = high_r.sample(10)
    return high_sample


def sample_med(med_risk_patients):
    med_r = med_risk_patients[med_risk_patients['appopen']>150]
    med_r = med_r[med_r['use_age']>50]
    med_sample = med_r.sample(10)
    return med_sample

def sample_low(low_risk_patients):
    low_r = low_risk_patients[low_risk_patients['appopen']>150]
    low_r = low_r[low_r['use_age']>50]
    low_sample = low_r.sample(10)
    return low_sample



def bar_plot(high_risk_patients, med_risk_patients):
    pf1 = high_risk_patients.max().to_frame()
    pf1 = pf1.T
    pf2 = med_risk_patients.max().to_frame()
    pf2 = pf2.T
    pf1 = pd.concat([pf1,pf2])
    pf1.replace({43189.0:1, 43178.0:2}, inplace=True)
    pf1 = pf1.drop(['user_id', 'index'], axis=1)
    pf1.plot.bar(pf1.index)
    plt.xticks( np.arange(2), ('High_Risk', 'Med_Risk') )
    plt.ylabel('Total Number of Acute/Critical Cases')


def plot_10_users_ivan(df,bg_values, df_s):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    user_ids = df_s.user_id.values

    for user_id in user_ids:
        sub_df = bg_values[bg_values.user_id == user_id]

        sub_df.sort_values(by = ['days_f0bg', 'bg'], axis=0, ascending=[True, True], inplace=True)
        sub_df = sub_df[sub_df.bg_appopen>0]
        sub_df= sub_df.dropna()
        sub_df['bg']= sub_df['bg']*1.0/sub_df['bg'][0:1].values

        x = sub_df.days_f0bg
        y = sub_df.bg

        #plt.scatter(x = x, y = y, c = np.random.rand(3,1))

        sub_df.plot(x = 'days_f0bg', y = 'bg', c = np.random.rand(3,1), ax = ax, kind = 'line', label = 'user {}'.format(user_id))

    plt.legend(loc='upper right', prop={'size':5})
    plt.xlabel("app age")
    plt.ylabel("BG Levels ")
    plt.title("BG Changes vs Time for Low Risk Users")

    plt.xlim(0,120)
    #plt.ylim(0,400)
    plt.show()


def churn_bg_entry(gf):
    l=[]
    for i in range(600):
        tot = gf[gf.tot_bg_open==i]['Churn'].sum()
        l.append((i,tot))
    l.remove(l[0])
    l = pd.DataFrame(l)
    plt.plot(l[0], l[1])
    plt.xlim(0,150)
    plt.ylim(0,1000)
    plt.xlabel("Num of BG Entries")
    plt.ylabel("Sum of Churn")
    plt.title("Churn vs BG-entry")
    plt.show()

def tables_without_oldusers(df, user_profiles):
    after_min_time = user_profiles[user_profiles.dateOfFirstLaunch>= open_by_user.Date.min()].reset_index()
    user_ids = after_min_time.user_id.unique()
    user_profiles= user_profiles.reset_index()
    usr_prf = user_profiles['user_id'].unique()

    df.drop(users_to_del, inplace=True)
    return df






if __name__ =='__main__':
    more_than_five = pd.read_csv('more_than_five.csv')
    user_profiles = pd.read_csv('user_profiles.csv')
    bg_values = pd.read_csv('bg_values.csv')
    bg_values = pd.read_csv('bg_values.csv')
    grouped_all = pd.read_csv('group_all_tables.csv')
    after_min_time = user_profiles[user_profiles.dateOfFirstLaunch>= open_by_user.Date.min()].reset_index()
    user_ids = after_min_time['user_id'].unique()
    # more_than_10 = more_than_five[more_than_five['appopen']>=10]
    # low_risk_patients, low_users = low_risk_users(more_than_10)
    # med_risk_patients, med_users = med_risk_users(more_than_10)
    # high_risk_patients, high_users = high_risk_users(more_than_10)
    # high_sample = sample_high(high_risk_patients)
    # med_sample = sample_med(med_risk_patients)
    # low_sample = sample_low(low_risk_patients)
    #plot_10_users_ivan(low_risk_patients, bg_values, low_sample)
    #plot_10_users_ivan(med_risk_patients,bg_values, med_sample)
    #plot_10_users_ivan(high_risk_patients,bg_values, high_sample)
    #churn_bg_entry(grouped_all)
    more_than_5 = tables_without_oldusers(more_than_five)
    bg_values = tables_without_oldusers(bg_values)
    user_profiles = tables_without_oldusers(user_profiles)
    grouped_all =tables_without_oldusers(grouped_all)
