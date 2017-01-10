import pandas as pd
import random
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
import datetime as dt
import numpy as np



def color_more_than_3(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val > .3 or val<-.3 else 'black'
    return 'color: %s' % color

def scatter_matrix_plot(df, core_features):


    colors = ['red' if ix else 'blue' for ix in df.Churn]
    scatter_matrix(df[core_features], figsize=(6,6), diagonal='hist', color=colors)

    ax = scatter_matrix(df[core_features], figsize= (30,30) ,diagonal ='kde',color=colors)
    [plt.setp(item.yaxis.get_majorticklabels(), 'size', 20) for item in ax.ravel()]
    [plt.setp(item.xaxis.get_majorticklabels(), 'size', 20) for item in ax.ravel()]

    [plt.setp(item.yaxis.get_label(), 'size', 20) for item in ax.ravel()]
    # #x labels
    [plt.setp(item.xaxis.get_label(), 'size', 20) for item in ax.ravel()]
    plt.show()

def scatter_plot_users_max_bg(bgdf):

    #this is not working
    max_bg = bgdf[bgdf.avg_bg>10]
    crit = (max_bg[(max_bg['avg_bg']> 350) | (max_bg['avg_bg']<50)])
    acc1 = max_bg[(max_bg['avg_bg']>=50) &(max_bg['avg_bg']<70)]
    acc2 = max_bg[(max_bg['avg_bg']>=250) &(max_bg['avg_bg']<350)]
    acc = pd.concat([acc1, acc2])
    norm = max_bg[(max_bg['avg_bg']<= 250) | (max_bg['avg_bg']>=70)]
    plt.axvline(x=350, c='salmon', linestyle = '--', label='critical limit')
    plt.axvline(x=50, c='salmon',linestyle = '--')
    plt.axvline(x=250, c='green', linestyle = '--', label='acute limit')
    plt.axvline(x=70, c='green',linestyle = '--')
    plt.axvline(x=150, c='blue', linestyle= '-', label = 'AVG BG')
    plt.legend()
    plt.xlim(0,600)
    plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') # labels along the bottom edge are off


    plt.scatter(crit.avg_bg,crit.user_id,  c='r', alpha=.9)
    plt.scatter(acc.avg_bg,acc.user_id,  c='r', alpha=.3)
    plt.scatter(norm.avg_bg ,norm.user_id, c= 'b', alpha=.1)



    plt.xlabel("Avg BG Level mg/dL")
    plt.ylabel("Users")
    plt.title("AVG BG Levels for all Users")

    plt.ylim(0,4000)

    plt.show()




def user_eng_bar_plot(bgdf):
    max_bg = bgdf[bgdf.max_bg>10]
    crit = (max_bg[(max_bg['max_bg']> 350) | (max_bg['min_bg']<50)])
    acc1 = max_bg[(max_bg['min_bg']>=50) &(max_bg['min_bg']<70)]
    acc2 = max_bg[(max_bg['max_bg']>=250) &(max_bg['max_bg']<350)]
    acc = pd.concat([acc1, acc2])
    norm = max_bg[(max_bg['max_bg']<= 250) | (max_bg['min_bg']>=70)]
    tot = crit.shape[0]+ acc.shape[0]+ norm.shape[0]
    max_bg = bgdf[bgdf.max_bg>10]
    crit = (max_bg[(max_bg['max_bg']> 350) | (max_bg['min_bg']<50)])
    acc1 = max_bg[(max_bg['min_bg']>=50) &(max_bg['min_bg']<70)]
    acc2 = max_bg[(max_bg['max_bg']>=250) &(max_bg['max_bg']<350)]
    acc = pd.concat([acc1, acc2])
    BG_NUMS = pd.DataFrame([[crit.shape[0]*100.0/tot, acc.shape[0]*100.0/tot, norm.shape[0]*100.0/tot]], columns=['High Risk', 'Med Risk', 'Low Risk'])
    BG_NUMS.plot.barh()
    plt.ylabel("Number of High/Med/Low Risk Users")
    plt.xlabel("Percentage")
    plt.title("Comparison of High/Med/Low Risk Users")
    #plt.show()
    return BG_NUMS





if __name__=='__main__':
    gf = pd.read_csv('group_all_tables.csv')
    #bgdf = pd.read_csv('bg_values.csv')
    bgdf = pd.read_csv('bg_grouped.csv')
    #user_eng_bar_plot(bgdf)
    scatter_plot_users_max_bg(bgdf)
    engagement_columns =  [u'appopen',
                       u'max_open/day',                      u'use_age',
                              u'Churn',                      u'1d_open',
                            u'2dacute',                      u'2d_crit',
                        u'2d_bg_entry',                    u'min_bg_2d',
                          u'max_bg_2d',                   u'1d_carbent',
                           u'1d_wtent',                      u'1w_open',
                            u'1w_crit',                      u'1wacute',
                        u'1w_bg_entry',                    u'min_bg_1w',
                          u'max_bg_1w',                   u'1w_carbent',
                           u'1w_wtent',                   u'tot_wt_ent',
                      u'max_wtent/day',
                        u'tot_carbent',
                             u'max_bg',                       u'avg_bg',
                       u'tot_critical',                    u'tot_acute',
                        u'tot_bg_open',                        u'>20_x']
    #scatter_matrix_plot(gf,engagement_column)

    # core_features = []
    # corr = grouped_add_1week.corr().style.applymap(color_more_than_3)
