# PHASE 4 PROJECT - CHICAGO CAR CRASHES



## BUSINESS UNDERSTANDING

*   **Context:** This project aims to analyze car accident data to identify the primary contributing causes. Understanding these causes is crucial for developing effective strategies to reduce traffic accidents, improve road safety, and ultimately save lives.
*   **Stakeholders:**
    *   **Vehicle Safety Board:** Interested in understanding vehicle-related factors and informing vehicle safety standards.
    *   **City of Chicago (or other municipalities):** Interested in optimizing traffic management, infrastructure planning, and public safety initiatives.
*   **Goals:**
    *   Develop a predictive model that accurately identifies the primary contributing cause of car accidents.
    *   Provide actionable insights to the Vehicle Safety Board and the City of Chicago to inform their efforts in reducing traffic accidents.
    *   Identify high-risk areas and demographic groups for targeted safety interventions.
    *   Evaluate the effectiveness of existing safety measures and guide the development of new strategies.

## PROBLEM STATEMENT

*   **The Challenge:** Traffic accidents are a significant public health concern, resulting in injuries, fatalities, and economic losses. Understanding the underlying causes of these accidents is essential for implementing effective prevention strategies.
*   **Specific Questions:**
    *   What are the most frequent primary contributing causes of car accidents?
    *   Which factors (e.g., driver behavior, road conditions, weather, vehicle-related issues) are most strongly associated with different accident causes?
    *   Can we accurately predict the primary contributing cause of an accident based on available data?
    *   Are there specific patterns or trends in accident causes based on location, time of day, weather conditions, or driver demographics?
*   **Objectives:**
    *   Build a classification model to predict the `PRIM_CONTRIBUTORY_CAUSE` from the available features.
    *   Achieve a high level of accuracy and F1-score, particularly for minority classes.
    *   Provide interpretable results that can be translated into actionable recommendations.

## OBJECTIVES
The main Objective of this dataset is to:

a) Develop a machine learning model that predicts the primary contributory cause of vehicle accidents with reasonable accuracy.

b)Identify the most significant factors influencing crashes, such as driver behavior, road conditions, weather, and vehicle attributes.

c) Determine which variables contribute most to accidents, such as reckless driving, speeding, distracted driving, or poor road conditions.


## DATA UNDERSTANDING
*   **Dataset Source:** The data used in this project comes from [Specify the data source, e.g., the City of Chicago Data Portal, a specific government agency, etc.]. Provide a link if available.
*   **Data Description:** The dataset contains information about car accidents, including:
    *   **Target Variable:** `PRIM_CONTRIBUTORY_CAUSE` (Primary Contributing Cause of the Accident)
    *   **Features:**
        *   Vehicle-related factors: `VEHICLE_TYPE`, `VEHICLE_DEFECT`, `VEHICLE_USE`, `VEHICLE_AGE_CATEGORY`
        *   Driver-related factors: `DRIVER_ACTION`, `DRIVER_VISION`, `DRIVER_CATEGORY`, `SEX`, `OCCUPANT_CNT`
        *   Environmental factors: `WEATHER_CONDITION`, `LIGHTING_CONDITION`, `ROADWAY_SURFACE_COND`, `ROAD_DEFECT`
        *   Location and Time: `POSTED_SPEED_LIMIT`, `TRAFFICWAY_TYPE`, `ALIGNMENT`, `CRASH_YEAR`, `CRASH_WEEKDAY`, `TIME_OF_DAY`, `CRASH_SEASON`
        *   Crash characteristics: `FIRST_CRASH_TYPE`, `TRAFFIC_CONTROL_DEVICE`, `DEVICE_CONDITION`
        *   Other: `TRAVEL_DIRECTION`
*   **Data Size:** The dataset contains approximately 1,267,020 rows and \[Number] columns.
*   **Data Quality Issues:**
    *   Missing values (represented as "UNKNOWN" or "NA" in many columns).
    *   Class imbalance in the target variable (`PRIM_CONTRIBUTORY_CAUSE`).
    *   Potential biases in the data collection process.
    *   Inconsistencies or errors in the data.

## EXPLORATORY DATA ANALYSIS FINDINGS



*   **Key Findings:**
    *   **Target Variable Distribution:** `DRIVER BEHAVIOR` and `OTHER/UNKNOWN` are the most frequent primary contributing causes.
      

    *   **Feature Distributions:** 
       




    *   **Relationships between Features and Target Variable:**
     ![image](https://github.com/user-attachments/assets/a573fb8d-f3bd-4ec2-ae38-451d81267e7e)
      ![image](https://github.com/user-attachments/assets/5589ee0b-c1fc-4f11-a013-7abf11102b8c)
      ![image](https://github.com/user-attachments/assets/c48efb87-b994-4eec-87ce-0ea236583fcc)








    

## Modeling

*   **Data Preprocessing:**
    *   **Handling Missing Values:** Imputed missing numerical values using the median and missing categorical values using the most frequent value.
    *   **Encoding Categorical Features:** Used One-Hot Encoding (with `handle_unknown='ignore'`) for categorical features to convert them into a numerical format suitable for machine learning models.
    *   **Feature Scaling:** Applied `StandardScaler` to scale numerical features.
    *   **Data Splitting:** Split the data into training, validation, and test sets (70/15/15) with stratified sampling to maintain class proportions.
*   **Model Selection:**
    *   **Baseline Model:** Logistic Regression (for initial comparison)
    *   **Primary Models:**
        *   Random Forest
        *   K-Nearest Neighbors (KNN)
        *   Gradient Boosting (Specifically the HistGradientBoostingClassifier)
*   **Addressing Class Imbalance:**
    *   Used `class_weight='balanced'` in Random Forest and Gradient Boosting to account for the imbalanced nature of the target variable.
    *   Considered using oversampling or undersampling techniques, but ultimately did not implement them due to concerns about overfitting.
*   **Hyperparameter Tuning:**
    *   Used `GridSearchCV` with 3-fold cross-validation to tune the hyperparameters of each model.
    *   Defined hyperparameter grids based on prior knowledge and experimentation.
*   **Model Training:**
    *   Trained each model on the training set using the best hyperparameters found by GridSearchCV.

## Evaluation

*   **Evaluation Metrics:**
    *   Accuracy (Overall performance)
    *   Precision (Minimize false positives, critical for targeted interventions)
    *   Recall (Minimize false negatives, for comprehensiveness)
    *   F1-score (Harmonic mean of precision and recall; primary metric due to class imbalance)
    *   Confusion Matrix (Detailed breakdown of prediction accuracy by class)
*   **Validation Results (from GridSearchCV):**
    *   Random Forest: Best F1 Score = 0.5577
        *   Best Parameters: {'classifier\_\_max\_depth': 10, 'classifier\_\_min\_samples\_leaf': 1, 'classifier\_\_min\_samples\_split': 2, 'classifier\_\_n\_estimators': 300}
    *   KNN: Best F1 Score = 0.5386
        *   Best Parameters: {'classifier\_\_n\_neighbors': 11, 'classifier\_\_p': 1, 'classifier\_\_weights': 'distance'}
    *   Gradient Boosting: Best F1 Score = 0.6025
        *   Best Parameters: {'classifier\_\_learning\_rate': 0.01, 'classifier\_\_max\_depth': 5, 'classifier\_\_n\_estimators': 200, 'classifier\_\_subsample': 0.8}
*   **Test Set Results:** 
Random Forest Test Set Performance:
Classification Report:
                            precision    recall  f1-score   support

            ALCOHOL/DRUGS       0.07      0.09      0.08      1128
              DISTRACTION       0.04      0.22      0.07      2745
          DRIVER BEHAVIOR       0.72      0.49      0.58     92756
            OTHER/UNKNOWN       0.61      0.48      0.54     79560
          ROAD CONDITIONS       0.28      0.35      0.31       736
                 SPEEDING       0.06      0.03      0.04       762
TRAFFIC SIGNALS VIOLATION       0.23      0.78      0.36      8071
           VEHICLE ISSUES       0.19      0.18      0.19      1193
                  WEATHER       0.12      0.75      0.21      3102

                 accuracy                           0.49    190053
                macro avg       0.26      0.38      0.26    190053
             weighted avg       0.62      0.49      0.53    190053

Confusion Matrix:
 [[  101    70   214   443     3     9   101    69   118]
 [   66   597   565  1012    10     3   181    66   245]
 [  368  6762 45079 21296   178   181 11022   176  7694]
 [  759  6955 15079 38407   466   175  9167   574  7978]
 [    2    19    87   276   260     1    27     6    58]
 [   23    37   160   226     1    25    80     3   207]
 [   14    74   792   565     4     6  6282     4   330]
 [   23    95   171   452     6     6    91   217   132]
 [   14     8   228   302     2     6   184    20  2338]]

KNN Test Set Performance:
Classification Report:
                            precision    recall  f1-score   support

            ALCOHOL/DRUGS       0.20      0.00      0.00      1128
              DISTRACTION       0.22      0.00      0.00      2745
          DRIVER BEHAVIOR       0.58      0.69      0.63     92756
            OTHER/UNKNOWN       0.55      0.52      0.53     79560
          ROAD CONDITIONS       0.33      0.00      0.00       736
                 SPEEDING       0.00      0.00      0.00       762
TRAFFIC SIGNALS VIOLATION       0.50      0.17      0.25      8071
           VEHICLE ISSUES       0.00      0.00      0.00      1193
                  WEATHER       0.28      0.02      0.04      3102

                 accuracy                           0.56    190053
                macro avg       0.29      0.16      0.16    190053
             weighted avg       0.54      0.56      0.54    190053

Confusion Matrix:
 [[    1     1   505   592     0     0    26     1     2]
 [    0     2  1511  1207     0     0    20     0     5]
 [    2     3 64398 27558     0     1   746     0    48]
 [    2     1 37641 41296     2     0   537     4    77]
 [    0     0   336   398     1     0     0     0     1]
 [    0     1   424   319     0     0    10     0     8]
 [    0     0  4482  2235     0     0  1348     0     6]
 [    0     0   551   628     0     1    10     0     3]
 [    0     1  1619  1399     0     0    25     0    58]]

 Gradient Boosting Test Set Performance:
Classification Report:
                            precision    recall  f1-score   support

            ALCOHOL/DRUGS       0.06      0.00      0.01      1128
              DISTRACTION       0.68      0.08      0.14      2745
          DRIVER BEHAVIOR       0.63      0.77      0.69     92756
            OTHER/UNKNOWN       0.62      0.55      0.58     79560
          ROAD CONDITIONS       0.34      0.11      0.17       736
                 SPEEDING       0.22      0.05      0.08       762
TRAFFIC SIGNALS VIOLATION       0.69      0.31      0.43      8071
           VEHICLE ISSUES       0.54      0.16      0.25      1193
                  WEATHER       0.32      0.09      0.14      3102

                 accuracy                           0.62    190053
                macro avg       0.46      0.24      0.28    190053
             weighted avg       0.62      0.62      0.61    190053

Confusion Matrix:
 [[    5     5   434   635     3     2    39     1     4]
 [    4   211  1327  1157     1     0    31     3    11]
 [   22    36 71003 20766    22    46   566    42   253]
 [   28    48 34518 43993   124    62   410    92   285]
 [    1     0   172   469    83     1     2     3     5]
 [    5     1   398   274     2    39    17     3    23]
 [    1     4  3937  1631     5     4  2474     0    15]
 [   14     1   417   552     1     2    10   192     4]
 [    5     4  1326  1419     4    21    17    21   285]]

 
 ![image](https://github.com/user-attachments/assets/7e34405d-72f7-48d6-aba1-74c5c303ad8e)

   ## *   Gradient Boosting: 
   ### **Strengths:**

 Highest overall accuracy and weighted F1-score.

* Reasonable performance on DRIVER BEHAVIOR and OTHER/UNKNOWN, which are the most frequent classes.

### **Weaknesses:**

 * Extremely poor performance on minority classes (ALCOHOL/DRUGS, ROAD CONDITIONS, SPEEDING, WEATHER, VEHICLE ISSUES, DISTRACTION). The precision and recall values are very low, indicating that the model is not able to identify these causes effectively.

* The confusion matrix shows that the model often misclassifies these minority classes as DRIVER BEHAVIOR or OTHER/UNKNOWN.

### **Interpretation and Recommendations:**

 * Gradient Boosting is the best-performing model, but it's still struggling with the class imbalance. Focus on improving its performance on the minority classes.

    *   Random Forest: 
### **Strengths:**

* High recall for WEATHER and TRAFFIC SIGNALS VIOLATION, meaning it captures a good proportion of these events.

### **Weaknesses:**

 * Extremely low precision for almost all classes. This means that many of the accidents it predicts as having a certain cause are actually due to something else.

 * The high recall combined with low precision suggests that the model is overgeneralizing and assigning a lot of accidents to these classes, even when they are not the true cause.

 * Lower F1-score, indicating that it's generally a poor model for this dataset.

### **Interpretation and Recommendations:**

#### * Random Forest is not performing well and is likely overfitting or not capturing the underlying patterns.    
    
   ## *   KNN:
### **Strengths:**

* None of significance based on the data.

### **Weaknesses:**

 * Extremely poor performance across the board, especially for minority classes.

 * Very low recall values, indicating that it's not able to identify the causes of accidents effectively.

 * High computation, especially for prediction with no comparable performance to other models.

### **Interpretation and Recommendations:**

* KNN is not suitable for this dataset. It's likely being overwhelmed by the high dimensionality and the large number of instances.

*   **Model Comparison:** Gradient Boosting outperformed Random Forest and KNN on the test set.

## Recommendations

*   **Based on the Model Results, we recommend the following to the City of Chicago and the Vehicle Safety Board:**
    1.  **Focus on driver education and awareness campaigns to address improper driving behaviors (targeting `DRIVER BEHAVIOR`):** 
    2.  **Investigate and reduce the ambiguity of crash data collection to better understand "Other/Unknown" causes:** 
    3.  **Implement targeted interventions for areas with high traffic signal violation rates (addressing `TRAFFIC SIGNALS VIOLATION`):** 
    4.  **Improve road maintenance during adverse weather conditions (addressing `ROAD CONDITIONS` and `WEATHER`):** 
    5.  **Enhance data collection:** Integrate more features.

## Future Work

*   **Address Class Imbalance More Aggressively:** Experiment with oversampling and undersampling techniques, as well as cost-sensitive learning.
*   **Feature Engineering:** Explore new features based on domain knowledge and interactions between existing features.
*   **Explore Other Models:** Consider other machine learning models, such as ensemble methods specifically designed for imbalanced data (e.g., EasyEnsemble, BalancedRandomForest).
*   **External Data:** Incorportate data.
*   **Geospatial Analysis:** Perform geospatial analysis to identify high-risk locations and patterns.
*   **Causal Inference:** Explore causal inference techniques to better understand the causal relationships between factors and accident causes.




