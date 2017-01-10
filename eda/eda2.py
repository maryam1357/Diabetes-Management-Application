import pandas as pd
import random
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import datetime as dt
import numpy as np


def med_risk_users(more_than_10):
    '''returns medium risk users'''

    g = more_than_10.copy()
    med_risk_patients = g[(g['tot_acute']> 0) & (g['tot_critical']==0)][['user_id','tot_acute','tot_critical','use_age','appopen']]
    med_users = med_risk_patients['user_id'].values.tolist()
    return med_risk_patients, med_users

def high_risk_users(more_than_10):
    '''returns high risk users'''

    g = more_than_10.copy()
    high_risk_patients = g[(g['tot_acute']> 1) & (g['tot_critical']>=1)][['user_id','tot_acute','tot_critical','use_age','appopen']]
    high_users = high_risk_patients['user_id'].values.tolist()
    return high_risk_patients, high_users

def low_risk_users(more_than_10):
    '''returns high risk users'''

    g = more_than_10.copy()
    low_risk_patients = g[(g['tot_acute']== 0) & (g['tot_critical']==0)][['user_id','tot_acute','tot_critical','use_age','appopen']]
    low_users = low_risk_patients['user_id'].values.tolist()
    return low_risk_patients, low_users

def sample_high(high_risk_patients):
    high_r = high_risk_patients[high_risk_patients['appopen']>150]
    high_r = high_r[high_r['use_age']>50]
    high_sample = high_r.sample(30)
    return high_sample

def sample_med(med_risk_patients):
    med_r = med_risk_patients[med_risk_patients['appopen']>150]
    med_r = med_r[med_r['use_age']>50]
    med_sample = med_r.sample(30)
    return med_sample

def sample_low(low_risk_patients):
    low_r = low_risk_patients[low_risk_patients['appopen']>150]
    low_r = low_r[low_r['use_age']>50]
    low_sample = low_r.sample(40)
    return low_sample

def bar_plot(high_risk_patients, med_risk_patients):

    '''bar plot of how engaged medium/ low/ high risk users are'''

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

def plot_sample_bgvalues(bg_values, df_s):
    '''plots sample bg values for critcial, acute, normal '''

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    user_ids = df_s.user_id.values

    for i, user_id in enumerate(user_ids):
        sub_df = bg_values[bg_values.user_id == user_id]

        sub_df.sort_values(by = ['days_f0bg', 'bg'], axis=0, ascending=[True, True], inplace=True)
        sub_df = sub_df[sub_df.bg_appopen>0]
        sub_df= sub_df.dropna()
        sub_df['bg']= sub_df['bg']*1.0/sub_df['bg'][0:1].values

        x = sub_df.days_f0bg
        y = sub_df.bg

        #plt.scatter(x = x, y = y, c = np.random.rand(3,1))

        sub_df.plot(x = 'days_f0bg', y = 'bg', c = np.random.rand(3,1), ax = ax, kind = 'scatter') #label = 'user {}'.format(i+1))

    #plt.legend(loc='upper right', prop={'size':5})
    plt.xlabel("Days Using App")
    plt.ylabel("BG Levels, Normalized")
    #plt.title("Low Risk Users BG Changes vs Time")

    plt.xlim(0,60)
    #plt.ylim(0,400)

def churn_bg_entry(gf):
    '''plots churn and acitivity based on bg entries
    INPUT: Grouped_all dataset'''

    l1=[]
    l2 =[]
    for i in range(600):
        tot = gf[gf.tot_bg_open==i]['Churn'].sum()
        l1.append((i,tot))
        tot2 = gf[gf.tot_bg_open==i]['is_active'].sum()
        l2.append((i,tot2))
    l1.remove(l1[0])
    l2.remove(l2[0])
    l1 = pd.DataFrame(l1)
    l2 = pd.DataFrame(l2)
    plt.plot(l1[0], l1[1], label='Churned Users')
    plt.plot(l2[0],l2[1], label='Active Users')
    plt.xlim(0,100)
    plt.ylim(0,200)
    plt.title("Activity and Churn vs BG-entry")
    plt.xlabel('Number of Blood Glucose Entries')
    plt.ylabel('Total Number of Users')
    plt.legend()
    plt.show()

def user_interaction_plot():
    '''Bar plot of how engaged users are'''

    data = (7092*100.0/31495+8575*100.0/31495 , 7190*100.0/31495 , 5057*100.0/31495, 3581*100.0/31495)
    index = np.arange(4)
    width = 0.5       # the width of the bars: can also be len(x) sequence

    rects = plt.bar(range(len(data)), data, width)
    plt.ylabel("Percentage")
    plt.title("How Often do Users Interact with the App? ")
    #plt.xlabel("Number of Interactions")
    plt.xticks(index+ .3 ,( '1-5 Times',  '6-20 Times', '21-100 Times', '101+ Times'))
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ =='__main__':
    more_than_five = pd.read_csv('more_thn_five.csv')
    user_profiles = pd.read_csv('user_profiles.csv')
    bg_values = pd.read_csv('bg_values.csv')
    grouped_all = pd.read_csv('group_all_tables.csv')
    grouped_all['is_active'] = (grouped_all.loc[:,'Churn']==0).astype(int)

    more_than_10 = more_than_five[more_than_five['appopen']>=10]
    low_risk_patients, low_users = low_risk_users(more_than_10)
    med_risk_patients, med_users = med_risk_users(more_than_10)
    high_risk_patients, high_users = high_risk_users(more_than_10)
    high_sample = sample_high(high_risk_patients)
    med_sample = sample_med(med_risk_patients)
    low_sample = sample_low(low_risk_patients)
    plot_10_users_ivan(bg_values, low_sample)
    plt.title("Low Risk Users BG Changes vs Time")
    plt.show()

    plot_sample_bgvalues(bg_values, med_sample)
    plt.title("Med Risk Users BG Changes vs Time")
    plt.show()

    #churn_bg_entry(grouped_all)
    #user_interaction_plot()
