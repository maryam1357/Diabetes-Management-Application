# Diabetes Management Application
### By Salma Riazi

## Summary

User engagement is a top concern for most mobile-app companies. It is specifically important for health care mobile apps, and critical for diabetes-tracking apps, as they are trying to drive behavior change. In this project I analyzed user engagement data, feature engineered new metrics
and predicted future engagement level. The main focus of this project was to predict churn, predict user engagement level, and analyze the drivers of engagement.

## Data

The data was provided by a top health tech company which I cannot name due to app engagement data confidentiality and HIPPA regulation for health data. It consists of a number of files containing user profile preferences, user interaction (e.g. blood glucose entry, weight entry, meal entry, app open) and exercise tracking for more than 100k users.

The main challenge with user engagement prediction is that only a small percentage of users actually use an app after the first day. This results in data imbalance (not as extreme as in fraud detection).

Another challenge with the data was that initially, as another side to this project, I was trying to predict user diabetes risk, and blood glucose fluctuations. Having done some research on diabetes, I realized that there are many features which  need to be tracked in order to come close to predicting risk based on blood glucose level alone. The important features that I was missing were: exact time of food intake, the kind of food, insulin injection, physical activity, stress level, and etc.

Because of data confidentiality, I have slightly replaced actual values for this report.

## Definitions

A few metrics will be defined and are used throughout the project.

### Churn:

Churn was defined to be inactivity for a 10-day period prior to the last day of available data. If a user is actively using a health tracker app, they usually open the app at least every day. Therefore, not being active for 10 days is a safe indicator that they have churned.

### Critical and Acute Diabetes:

Critical -  blood glucose less than 50 ml/dL and more than 350

Acute -  blood glucose less than 70 ml/dL and more than 250

## Exploratory Analysis

Exploratory analysis shows how user engagement behaves with passing time. 50% of users only use the app 1-5 times.

![Total User Engagement](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/total_eng.png)


I classified user engagement into 4 classes as seen in the table below:

![User Engagement Classification](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/class_table.png)

First week when users get the app, they are very excited, leading to high interactions during that week.

![User Engagement - 1 month after initial use](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/first_week.png)


I calculated user engagement one month after users signed up, for a 1-week period. 64% of users opted out after 1 month. Here are the results:

![User Engagement - 1 month after initial use](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/1month.png)


After 3 months user engagement decreases to only 15%.
![User Engagement - 3 months after initial use](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/3months.png)


Looking at the total number of users who churned and the total blood glucose entries, it seems like after 10 entries, the total churned users decrease from 2000 to 150.

![User Engagement - Churn and bg entry use](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/churn_bg_entr.png)

## Features

The features used in this project were:

1. User interaction with the app:

    - Opening the app

    - Entering a value (blood glucose, weight, meal intake, exercise)

2. User profiles

    - Diabetes type

    - App setting preferences

### Feature Engineering - Defining New Metrics

In order to make meaningful predictions and find the driving factors of user engagement for Diabetes Management App, new metrics had to be defined and calculated:

- Session Interval

- First 2 days user interaction

- First week user interaction

- Total interaction per day

- 1 week period interaction after 1 month being a user

- 1 week period interaction after 3 months being a user

- Blood glucose minimum and maximum for each user

- Blood glucose increase and decrease trends every 2 week period.

- Number of critical and acute cases for each user

- Risk levels for users:

    Low Risk: when a user has no acute or critical cases

    Medium Risk: when a user has no critical but at least 1 acute case

    High Risk: when a user has critical cases

    I looked at 30 random users from each category, normalized their blood glucose levels and plotted them with respect to the days they have been using the app to see if app usage helps keep blood glucose levels stable.

![Low Risk Users - BG values](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/low.png)

![Medium Risk Users - BG values](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/med.png)

![High Risk Users - BG values](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/high.png)

Looking at these plots, it cannot be concluded that the app has an effect. Further investigation is required.


## Churn Prediction

### Method

In order to predict churn, I used the newly defined metrics along with the profile preference features. I used SMOTE method to deal with class imbalance (more than 70% of total users had churned).

k-means clustering was used to cluster similar users in terms of user profiles and user behavior.

The classifier models I used were:

- Logistic Regression

- Random Forest Classifier

- Gradient Boosting Classifier

- Ensemble (Majority vote) Classifier

Here is the Receiver Operating Characteristic curve for the models used.

![ROC CURVE](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/ROC.png)



As seen in the ROC curve, Random Forest is the best classifier for this purpose. The confusion matrix also shows how well the model is behaving. For churn prediction, a high recall and a low false negative is desirable. In this case false negative would be if churn is not predicted while user churns.

![Confusion Matrix](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/Conf_mat.png)

## Engagement Class Prediction

As mentioned earlier, I classified users engagement levels into 4 classes. Using the new defined metrics and user profile features, I predicted users engagement level one month after first getting the app.

![User Engagement Classification](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/class_table.png)

For this multi-class classifier, I used undersampling for class imbalance, which resulted in better accuracy compared to oversampling.

I used the same models as in previous section, and again Random Forest was the best classifier.

## Important Features - Engagement Drivers

The classifiers ran for this projects agreed on the important features which were:

- First week app open

- Session Interval

- First day app open

- First week blood glucose entries

- First day blood glucose entries

- Maximum number of times user opened the app in a day


Following plot compares some of these values for active vs churned users.

![Important Features](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/important_feat.png)

This histogram compares first-week interaction of active and churned users.

![first_week](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/1week_interaction.png)

## Recommendations

In order to measure success, quantifiable metrics are needed. The recommendation for the company is to set a goal of for example getting 25 app interactions on the first week. To get to the recommended number of app interaction, marketing strategies are needed. Maybe have a reward system that users can collect points every time they enter something in the app.

Another way is to connect similar users to each other in terms of user behavior, age, diabetes type, weight, and have them compete in managing their diabetes.

## Next Steps

The next step is to get more user data (age, weight, health background, etc.) and cluster similar users in order to connect them to each other, or recommend similar-user tips and conduct A/B testing.
