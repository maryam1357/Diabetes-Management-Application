import random
import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from itertools import combinations, izip

columns_kmeans = [u'max_open/day',
 u'1d_open',
 u'2dacute',
 u'2d_crit',
 u'2d_bg_entry',
 u'min_bg_2d',
 u'max_bg_2d',
 u'1d_carbent',
 u'1d_wtent',
 u'1w_open',
 u'1w_crit',
 u'1wacute',
 u'1w_bg_entry',
 u'min_bg_1w',
 u'max_bg_1w',
 u'1w_carbent',
 u'1w_wtent',
 u'min_bg',
 u'max_bg',
 u'avg_bg',
 u'tot_critical',
 u'tot_acute',
 u'tot_bg_open',
 u'BG_Schedule',
 u'dateOfFirstLaunch',
 u'Meal_Reminders_Set',
 u'TrackingExerciseMinutes',
 u'Total_BG_Reminders',
 u'carbBudget',
 u'diabetes.type_Gestational',
 u'diabetes.type_LADA',
 u'diabetes.type_Other',
 u'diabetes.type_Pre-diabetic',
 u'diabetes.type_Prediabetes',
 u'diabetes.type_Type I',
 u'diabetes.type_Type II',
 u'gender_Female',
 u'gender_Male',
 u'CarbEntryPreference_estimate',
 u'CarbEntryPreference_numeric',
 u'UserTreatmentType_Both',
 u'UserTreatmentType_Meds',
 u'1w_eng']

df =pd.read_csv('more_than_five.csv')
def kmeans(df):
    new_df_for_kmeans = df.copy().loc[:,columns_kmeans]
    loc = []
    for row in new_df_for_kmeans.index:
        new_row = new_df_for_kmeans.loc[row,:]
        if all(new_row.isnull()) == False and (new_row[0] > .1 or new_row[0] < -.1):
            # if new_row[0] > .1 or new_row[0] < -.1:
            loc.append(True)
        else:
            loc.append(False)
    new_df_for_kmeans2 = new_df_for_kmeans.copy()[loc]

    # clusters = []
    # for k in range(1, 10):
    #     model3 = KMeans(n_clusters=k)
    #     model3.fit(new_df_for_kmeans2)
    #     clusters.append(model3.cluster_centers_)
    #
    #
    #
    # K = range(1,50)
    # KM = [KMeans(n_clusters=k).fit(new_df_for_kmeans2) for k in K]
    # centroids = [k.cluster_centers_ for k in KM]
    #
    # D_k = [cdist(new_df_for_kmeans2, cent, 'euclidean') for cent in centroids]
    # cIdx = [np.argmin(D,axis=1) for D in D_k]
    # dist = [np.min(D,axis=1) for D in D_k]
    # avgWithinSS = [sum(d)/new_df_for_kmeans2.shape[0] for d in dist]
    #
    # # Total with-in sum of square
    # wcss = [sum(d**2) for d in dist]
    # tss = sum(pdist(new_df_for_kmeans2)**2)/new_df_for_kmeans2.shape[0]
    # bss = tss-wcss
    #
    # kIdx = 10-1
    #
    # # elbow curve
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(K, avgWithinSS, 'b*-')
    # ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
    # markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    # plt.grid(True)
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Average within-cluster sum of squares')
    # plt.title('Elbow for KMeans clustering')
    # plt.show() # 5 is good for k
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(K, bss/tss*100, 'b*-')
    # plt.grid(True)
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Percentage of variance explained')
    # plt.title('Elbow for KMeans clustering')

    model3 = KMeans(n_clusters=10)
    new_df_for_kmeans2['cluster'] = model3.fit_predict(new_df_for_kmeans2)


    clusters = model3.cluster_centers_
    for cluster in clusters:
        folium.Marker([cluster[0], cluster[1]]).add_to(map)

    new_df['cluster'] = new_df_for_kmeans2['cluster']
