# NY Air Quality Prediction

This repository contains a project aimed at predicting air quality in New York City based on environmental data. The project utilizes machine learning techniques to analyze the dataset and create predictive models that can forecast air quality levels.

## Dataset

The dataset used in this project is the **New York City Air Quality** dataset, which is available on Kaggle. It contains data about air quality measurements taken from various sensors around the city, including pollutant levels and environmental conditions.

You can access the dataset [here on Kaggle](https://www.kaggle.com/datasets/fatmanur12/new-york-air-quality/data).

### Dataset Features:
- Categories of topics such as: Emissions, General Pollution, etc.
- Geographical locations
- Time Period
- Start Date
- Air Quality Category

## Libraries and Tools

The project makes use of the following libraries and tools:

- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical operations and array handling
- **Matplotlib & Seaborn**: For data visualization
- **Scikit-learn**: For implementing machine learning models and evaluation
- **Jupyter Notebooks**: For running and documenting the process interactively

## Learning Models

The project explores various machine learning models, including:

- **K-Nearest Neighbors (KNN)**: A non-parametric method used for classification and regression tasks. It predicts the value based on the 'k' nearest neighbors in the feature space. This method was chosen due to its simplicity and effectiveness for this type of problem.

The model is evaluated using performance metrics such as **Accuracy Score** and **Classification Report**.

## Process Overview

### 1. Data Collection and Preprocessing
- The data was collected from the Kaggle dataset linked above.
- No **Missing values** were found.
- Non numerical variables were **Encoded** with **LabelEncoder** into numerical ones.

### 2. Exploratory Data Analysis (EDA)
- The dataset was thoroughly analyzed to understand its structure, distributions, and relationships.
- Visualizations such as bar plots and heatmap were used to gain insights into air quality trends.

### 3. Model Training
- KNeighborsClassifier learning model were trained on the dataset.
- **Cross-validation** was used to assess model performance and reduce overfitting.

### 4. Model Evaluation
- The trained models were evaluated on a separate test set.
- Metrics such as **Accuracy Score** and **Classification Report** were used to assess each model's performance.

## Results

After training and evaluating several models, the following results were obtained:

- The **Classification Report** indicates that the KNN model is performing extremely well, with all precision, recall and f1-score values ​​close to 1.00, which is excellent. Furthermore, the total accuracy in the report is 1.00 (100% accuracy).
- The **Accuracy**  metric (99.65%) is slightly lower than the accuracy provided in the classification report.

The classification report is the most complete evaluation method and seems to indicate that the model is performing exceptionally well.

## Conclusion

This project successfully predicted air quality levels in New York City using machine learning model. By analyzing data on geolocation and environmental conditions, the models were able to identify key patterns and predict future air quality with reasonable accuracy. Future improvements could include tuning models further, incorporating more external factors, or using deep learning models for enhanced predictions.