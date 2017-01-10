import pandas as pd
import random
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import numpy as np



def hist_plot_user_engagement(df):
    '''histogram of first week, first 2 days and total app opens '''
    x_ticks = np.arange(0,325,25)
    x = df['appopen']
    y = df['1w_open']
    z = df['1d_open']
    bins = 400
    plt.hist(x, bins, alpha=0.5, label='Total')
    plt.hist(y, bins, alpha=0.5, label='First Week')
    plt.hist(z, bins, alpha=.5, label='First 2 Days')

    plt.legend(loc='upper right')
    plt.xlabel("Num of Times App Opened")
    plt.ylabel("Num of Users")
    plt.title("User App Engagement")

    plt.ylim(0,1000)

    plt.xlim(0 , 325)

    plt.xticks(x_ticks)



    '''df['1w_open'].sum()/df['appopen'].sum() =26%
    26 percent of total Engagement is over the first week'''

def get_one_user_bg(userid, bg_values):
    df_temp = bg_values[bg_values['user_id']==userid]
    return df_temp

def get_low_risk_users(bg_values):
    s = bg_values[bg_values['acute']==0]
    sh = bg_values[bg_values['critical']==0]
    s = s.set_index('user_id')
    sh = sh.set_index('user_id')
    sh.drop(['Date','critical', 'acute'], axis =1)
    ch1 = pd.concat([s,sh], axis=1, join='inner')




if __name__=='__main__':
    df = pd.read_csv('more_thn_five.csv')


    #hist_plot_user_engagement(df)
    box_plot(df)
    plt.show()
    #churn_bg_entry(gf)
