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
    *   **Feature Distributions:** [Summarize the distributions of key features and highlight any interesting patterns or anomalies.]
    *   **Relationships between Features and Target Variable:** [Describe the relationships between key features and the `PRIM_CONTRIBUTORY_CAUSE`. Use specific examples, e.g., "Accidents involving `VEHICLE_TYPE_MOTORCYCLE` are more likely to be attributed to `DRIVER BEHAVIOR`."]
    *   **Missing Value Analysis:** [Describe the extent of missing values in each column and how they were handled.]

*   **Visualizations:** [Include key visualizations generated during EDA, such as bar plots of categorical feature distributions, box plots of numerical features vs. target variable, and correlation heatmaps.  You can link to the images here if they are stored separately.]

## 5. Modeling

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

## 6. Evaluation

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
*   **Test Set Results:** \[Paste the classification reports and/or confusion matrices for each model here.]
    *   Gradient Boosting: \[Report metrics and discuss key findings, e.g., strengths, weaknesses, and misclassification patterns.]
    *   Random Forest: \[Report metrics and discuss key findings.]
    *   KNN: \[Report metrics and discuss key findings. If performance is very poor, state that the model is not suitable for the data.]
*   **Model Comparison:** Gradient Boosting outperformed Random Forest and KNN on the test set.

## 7. Recommendations

*   **Based on the Model Results, we recommend the following to the City of Chicago and the Vehicle Safety Board:**
    1.  **Focus on driver education and awareness campaigns to address improper driving behaviors (targeting `DRIVER BEHAVIOR`):** 
    2.  **Investigate and reduce the ambiguity of crash data collection to better understand "Other/Unknown" causes:** 
    3.  **Implement targeted interventions for areas with high traffic signal violation rates (addressing `TRAFFIC SIGNALS VIOLATION`):** 
    4.  **Improve road maintenance during adverse weather conditions (addressing `ROAD CONDITIONS` and `WEATHER`):** 
    5.  **Enhance data collection:** Integrate more features.

## 8. Future Work

*   **Address Class Imbalance More Aggressively:** Experiment with oversampling and undersampling techniques, as well as cost-sensitive learning.
*   **Feature Engineering:** Explore new features based on domain knowledge and interactions between existing features.
*   **Explore Other Models:** Consider other machine learning models, such as ensemble methods specifically designed for imbalanced data (e.g., EasyEnsemble, BalancedRandomForest).
*   **External Data:** Incorportate data.
*   **Geospatial Analysis:** Perform geospatial analysis to identify high-risk locations and patterns.
*   **Causal Inference:** Explore causal inference techniques to better understand the causal relationships between factors and accident causes.




