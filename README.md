# Diabetes Management Application
### By Salma Riazi

## Summary

User engagement is a top concern of most mobile-app making companies. It is specifically important for health care mobile apps, and even more important for diabetes-tracking apps, because they are trying to drive behavior change on the users. In this project I analyzed user engagement data, feature engineered new metrics
and predicted future engagement level. The main focus of this project was to predict churn, predict user engagement level, and analyze the drivers of engagement.

## Data

The data was provided by a top health tech company which I cannot name due to app engagement data confidentiality and HIPPA regulation for health data. It consists of a number of files containing user profile preferences, user interaction (e.g. blood glucose entry, weight entry, meal entry, app open) and exercise tracking for more than 100k users.

The main challenge with user engagement prediction is that only a small percentage of users actually use an app after the first day. This results in data imbalance (not as extreme as in fraud detection).

Another challenge with the data was that initially, as another side to this project, I was trying to predict user diabetes risk, and blood glucose fluctuations. Having done some research on diabetes, I realized that there are many features which  need to be tracked in order to come close to predicting risk based on blood glucose level. The important features that I needed but did not have for this purpose were exact time of food intake, the kind of food, insulin injection, physical activity, stress level, and etc.


## Exploratory Analysis

Exploratory analysis shows how user engagement behaves with passing time.

![Total User Engagement](https://github.com/salmariazi/predicting-seizures/blob/master/figures/interictal.png)
