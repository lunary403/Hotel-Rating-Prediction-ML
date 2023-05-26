<!DOCTYPE html>
<html>
<body>
  <h1>Hotel Rating Prediction</h1>
  <p>This is a machine learning project that focuses on predicting hotel ratings based on various features and reviews. The goal is to build and evaluate different models to accurately predict the rating given by reviewers.</p>
  
  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#project-overview">Project Overview</a></li>
    <li><a href="#data-preprocessing">Data Preprocessing</a></li>
    <li><a href="#model-training-and-evaluation">Model Training and Evaluation</a></li>
    <ul>
      <li><a href="#logistic-regression">Logistic Regression</a></li>
      <li><a href="#decision-tree-classifier">Decision Tree Classifier</a></li>
      <li><a href="#random-forest-classifier">Random Forest Classifier</a></li>
      <li><a href="#k-nearest-neighbors-classifier">K-Nearest Neighbors Classifier</a></li>
    </ul>
    <li><a href="#regression-models">Regression Models</a></li>
    <ul>
      <li><a href="#linear-regression">Linear Regression</a></li>
      <li><a href="#k-nearest-neighbors-regression">K-Nearest Neighbors Regression</a></li>
      <li><a href="#random-forest-regression">Random Forest Regression</a></li>
      <li><a href="#gradient-boosting-regression">Gradient Boosting Regression</a></li>
    </ul>
  </ul>
  
  <h2>Project Overview</h2>
  <p>The project aims to predict hotel ratings based on a dataset containing various features and reviews. The dataset is preprocessed to handle missing values, encode categorical variables, and perform feature scaling. The project includes both classification and regression models to cover different aspects of rating prediction.</p>
  
  <h2>Data Preprocessing</h2>
  <p>The data preprocessing steps involve handling outliers, encoding categorical variables, extracting relevant information from tags, and scaling numerical features. Outliers in certain columns are winsorized to mitigate their impact on the models. Categorical variables are encoded, and tags are extracted and organized into categories. Numerical features are scaled using the Min-Max scaling technique to ensure consistency across different ranges.</p>
  
  <h2>Model Training and Evaluation</h2>
  
  <h3>Logistic Regression</h3>
  <p>A logistic regression model is trained to predict hotel ratings. The top features with the highest correlation to the target variable are selected for training the model. The model's performance is evaluated using validation and test accuracy scores. The trained logistic regression model is saved for future use.</p>
  
  <h3>Decision Tree Classifier</h3>
  <p>A decision tree classifier is trained to predict hotel ratings. The same top features selected for logistic regression are used as input features. The model's performance is evaluated using validation and test accuracy scores. The trained decision tree classifier is saved for future use.</p>
  
  <h3>Random Forest Classifier</h3>
  <p>A random forest classifier is trained to predict hotel ratings. The model uses 150 estimators, a maximum depth of 25, and minimum samples per leaf of 75. The model's performance is evaluated using validation and test accuracy scores. The trained random forest classifier is saved for future use.</p>
  
  <h3>K-Nearest Neighbors Classifier</h3>
  <p>A k-nearest neighbors classifier is trained to predict hotel ratings. The model uses 21 neighbors for classification. The model's performance is evaluated using validation and test accuracy scores. The trained k-nearest neighbors classifier is saved for future use.</p>
  
  <h2>Regression Models</h2>
  
  <h3>Linear Regression</h3>
  <p>A linear regression model is trained to predict hotel ratings. The mean squared error (MSE) is calculated for both the training and validation sets to evaluate the model's performance.</p>
  
  <h3>K-Nearest Neighbors Regression</h3>
  <p>A k-nearest neighbors regression model is trained to predict hotel ratings. The mean squared error (MSE) is calculated for both the training and validation sets to evaluate the model's performance.</p>
  
  <h3>Random Forest Regression</h3>
  <p>A random forest regression model is trained to predict hotel ratings. The mean squared error (MSE) is calculated for both the training and validation sets to evaluate the model's performance.</p>
  
  <h3>Gradient Boosting Regression</h3>
  <p>A gradient boosting regression model is trained to predict hotel ratings. The mean squared error (MSE) is calculated for both the training and validation sets to evaluate the model's performance.</p>
  
</body>
</html>
