# Diabetes Management Application
### By Salma Riazi

## Summary

User engagement is a top concern of most mobile-app making companies. It is specifically important for health care mobile apps, and even more important for diabetes-tracking apps, because they are trying to drive behavior change on the users. In this project I analyzed user engagement data, feature engineered new metrics
and predicted future engagement level. The main focus of this project was to predict churn, predict user engagement level, and analyze the drivers of engagement.

## Data

The data was provided by a top health tech company which I cannot name due to app engagement data confidentiality and HIPPA regulation for health data. It consists of a number of files containing user profile preferences, user interaction (e.g. blood glucose entry, weight entry, meal entry, app open) and exercise tracking for more than 100k users.

The main challenge with user engagement prediction is that only a small percentage of users actually use an app after the first day. This results in data imbalance (not as extreme as in fraud detection).

Another challenge with the data was that initially, as another side to this project, I was trying to predict user diabetes risk, and blood glucose fluctuations. Having done some research on diabetes, I realized that there are many features which  need to be tracked in order to come close to predicting risk based on blood glucose level. The important features that I needed but did not have for this purpose were exact time of food intake, the kind of food, insulin injection, physical activity, stress level, and etc.

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

![Total User Engagement](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/1month_after.png)

I classified user engagement into 4 classes as seen in the table below:

![User Engagement Classification](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/class_table.png)

I calculated user engagement one month after users signed up, for a 1-week period. 64% of users opted out after 1 month. Here are the results:

![User Engagement - 1 month after initial use](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/1month_after.png)

After 3 months user engagement decreases to only 15%.
![User Engagement - 3 months after initial use](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/3month_after.png)


Looking at the total number of users who churned and the total blood glucose entries, it seems like after 10 entries, the total churned users decrease from 2000 to 150.
![User Engagement - 3 months after initial use](https://github.com/salmariazi/Diabetes_Monitor/blob/master/figures/churn_bg_entr.png)

### Features used

The features used in this project were:

1. User interaction with the app:

    - Opening the app

    - Entering a value (blood glucose, weight, meal intake, exercise)

2. User profiles

    - Diabetes type

    - App setting preferences

## Feature Engineering - Defining New Metrics

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
