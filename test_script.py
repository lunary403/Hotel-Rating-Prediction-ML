import pandas as pd
from sklearn.metrics import accuracy_score
import pickle

from sklearn.preprocessing import MinMaxScaler

from phase2 import winsorize_column, encode_reviews, extract_tags, tripType_encoding, extract_country, \
    winsorize_outliers


def preprocess_data(df):
    columns_to_winsorize = [
        'Additional_Number_of_Scoring',
        'Average_Score',
        'Review_Total_Negative_Word_Counts',
        'Total_Number_of_Reviews',
        'Review_Total_Positive_Word_Counts',
        'Total_Number_of_Reviews_Reviewer_Has_Given',
        'lat',
        'lng',
    ]

    for column in columns_to_winsorize:
        df[column] = winsorize_column(df, column)

    df['lat'].fillna(df['lat'].mode()[0], inplace=True)
    df['lng'].fillna(df['lng'].mode()[0], inplace=True)

    df['days_since_review'] = df['days_since_review'].str.replace("[days]", '').astype(int)
    df['Review_Date'] = pd.to_datetime(df['Review_Date'])
    df['day'] = df['Review_Date'].dt.day
    df['month'] = df['Review_Date'].dt.month
    df['year'] = df['Review_Date'].dt.year
    df['day_of_week'] = df['Review_Date'].dt.day_name()
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['day_of_week'] = df['day_of_week'].map(day_map)

    df['Positive_Review'] = df['Positive_Review'].apply(encode_reviews, exclude_list=['No Positive', 'Nothing'])
    df['Negative_Review'] = df['Negative_Review'].apply(encode_reviews,
                                                        exclude_list=['No Negative', 'Nothing', 'nothing', 'N A'])
    df['Hotel_Name'] = df['Hotel_Name'].str.replace('H tel', 'Hotel')
    tags_df = df['Tags'].apply(lambda x: pd.Series(extract_tags(x)))
    df = pd.concat([df, tags_df], axis=1)
    df['trip_type'] = df['trip_type'].fillna('Leisure trip')
    df['trip_type'] = df['trip_type'].apply(tripType_encoding)
    df['room_type'] = df['room_type'].fillna('Double Room')
    df['device_type'] = df['device_type'].fillna('Submitted from a mobile device')
    df['Hotel_Address'] = df['Hotel_Address'].apply(extract_country)
    df['Hotel_Country'] = df['Hotel_Address']
    df['Reviewer_Nationality'] = df['Reviewer_Nationality'].str.split().str[-1]
    df['Similar_country'] = df.apply(lambda row: 1 if row['Hotel_Address'] == row['Reviewer_Nationality'] else 0,
                                     axis=1)
    df['stay_duration'] = df['stay_duration'].str.replace("[Stayed, night,s]", '')
    mode_duration = df['stay_duration'].mode()[0]
    df['stay_duration'] = df['stay_duration'].fillna(mode_duration)
    df['stay_duration'] = df['stay_duration'].astype(int)
    df['stay_duration'] = winsorize_outliers(df['stay_duration'])

    columns_to_scale = ['Additional_Number_of_Scoring', 'Average_Score', 'Negative_Review',
                        'Review_Total_Negative_Word_Counts', 'Total_Number_of_Reviews', 'Positive_Review',
                        'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews_Reviewer_Has_Given',
                        'days_since_review', 'lat', 'lng', 'month', 'year', 'day_of_week', 'trip_type',
                        'stay_duration', 'Similar_country']

    scaler = MinMaxScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    rev_map = {'High_Reviewer_Score': 1, 'Intermediate_Reviewer_Score': 0, 'Low_Reviewer_Score': -1}
    df['Reviewer_Score'] = df['Reviewer_Score'].replace(rev_map).astype(int)

    return df


# Load the preprocessed test data
test_df = pd.read_csv("hotel-tas-test-classification.csv")

# Preprocess the test data
test_df = preprocess_data(test_df)

correlation_matrix = test_df.corr()
target_correlation = correlation_matrix['Reviewer_Score']
sorted_correlation = target_correlation.abs().sort_values(ascending=False)

# Select the top features
top_features = sorted_correlation[1:6].index.tolist()

# Update feature matrix
X_test = test_df[top_features].to_numpy()
y_test = test_df['Reviewer_Score'].to_numpy()

# Load the Logistic Regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    clf = pickle.load(f)

# Predict using Logistic Regression
y_test_pred_lr = clf.predict(X_test)

# Load the Random Forest Classifier model
with open('random_forest_model.pkl', 'rb') as f:
    RFC = pickle.load(f)

# Predict using Random Forest Classifier
y_test_pred_rf = RFC.predict(X_test)

# Load the Decision Tree Classifier model
with open('decision_tree_model.pkl', 'rb') as f:
    DTC = pickle.load(f)

# Predict using Decision Tree Classifier
y_test_pred_dt = DTC.predict(X_test)

# Load the K-Nearest Neighbors Classifier model
with open('knn_model.pkl', 'rb') as f:
    KNC = pickle.load(f)

# Predict using K-Nearest Neighbors Classifier
y_test_pred_kn = KNC.predict(X_test)


# Calculate test accuracies
test_accuracy_lr = accuracy_score(y_test, y_test_pred_lr)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
test_accuracy_dt = accuracy_score(y_test, y_test_pred_dt)
test_accuracy_kn = accuracy_score(y_test, y_test_pred_kn)

# Print test accuracies
print("Test accuracy for Logistic Regression:", test_accuracy_lr)
print("Test accuracy for Random Forest Classifier:", test_accuracy_rf)
print("Test accuracy for Decision Tree Classifier:", test_accuracy_dt)
print("Test accuracy for K-Nearest Neighbors Classifier:", test_accuracy_kn)
print(test_df.info)
