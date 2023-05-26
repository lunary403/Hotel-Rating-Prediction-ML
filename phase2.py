import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def winsorize_outliers(data, iqr_multiplier=1.5):
    # Calculate the quartiles and IQR
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1

    # Calculate the upper and lower whisker values
    upper_whisker = q3 + (iqr_multiplier * iqr)
    lower_whisker = q1 - (iqr_multiplier * iqr)

    # Winsorize the data
    data = np.where(data > upper_whisker, upper_whisker, data)
    data = np.where(data < lower_whisker, lower_whisker, data)

    return data


def winsorize_column(df, column):
    df[column] = winsorize_outliers(df[column])
    return df[column]


def encode_reviews(value, exclude_list):
    for exclude_string in exclude_list:
        if exclude_string in value:
            return 0
    return 1


def extract_tags(tags_str):
    # Remove square brackets, single quotes, and split the string into a list of tags
    tags = tags_str.strip("[]").replace("'", "").split(", ")

    # Create an empty dictionary to store the tag categories
    tag_categories = {}

    # Iterate over each tag in the list
    for tag in tags:
        # Check if the tag contains "trip"
        if "trip" in tag:
            # Assign the tag value to the "trip_type" category in the dictionary
            tag_categories["trip_type"] = tag.strip()
        # Check if the tag contains "Room"
        elif "Room" in tag:
            # Assign the tag value to the "room_type" category in the dictionary
            tag_categories["room_type"] = tag.strip()
        # Check if the tag contains "Stayed"
        elif "Stayed" in tag:
            # Assign the tag value to the "stay_duration" category in the dictionary
            tag_categories["stay_duration"] = tag.strip()
        # Check if the tag contains "Submitted"
        elif "Submitted" in tag:
            # Assign the tag value to the "device_type" category in the dictionary
            tag_categories["device_type"] = tag.strip()
        # If none of the above conditions are met, assume it's a "group_type" tag
        else:
            # Assign the tag value to the "group_type" category in the dictionary
            tag_categories["group_type"] = tag.strip()

    # Return the dictionary containing the organized tag categories
    return tag_categories


def tripType_encoding(value):
    exclude_list = ['Leisure trip']
    for exclude_string in exclude_list:
        if exclude_string in value:
            return 1
    return 0


def extract_country(address):
    # Split the address by spaces
    address_parts = address.split()
    # Get the last element as the country
    country = address_parts[-1]
    return country


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

    df['lat'].fillna(df['lat'].mode(), inplace=True)
    df['lng'].fillna(df['lng'].mode(), inplace=True)

    df['days_since_review'] = df['days_since_review'].str.replace("[days]", '').astype(int)
    df['Review_Date'] = pd.to_datetime(df['Review_Date'])
    df['day'] = df['Review_Date'].dt.day
    df['month'] = df['Review_Date'].dt.month
    df['year'] = df['Review_Date'].dt.year
    df['day_of_week'] = df['Review_Date'].dt.day_name()
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['day_of_week'] = df['day_of_week'].map(day_map)
    df = df.drop(['Review_Date', 'day'], axis=1)

    df['Positive_Review'] = df['Positive_Review'].apply(encode_reviews, exclude_list=['No Positive', 'Nothing'])
    df['Negative_Review'] = df['Negative_Review'].apply(encode_reviews,
                                                        exclude_list=['No Negative', 'Nothing', 'nothing', 'N A'])
    df['Hotel_Name'] = df['Hotel_Name'].str.replace('H tel', 'Hotel')
    tags_df = df['Tags'].apply(lambda x: pd.Series(extract_tags(x)))
    df = pd.concat([df, tags_df], axis=1)
    df.drop('Tags', axis=1, inplace=True)
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


df = pd.read_csv("hotel-classification-dataset.csv")

df_shuffled = df.sample(n=len(df), random_state=1)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)
train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=1)

# Preprocess the data
train_df = preprocess_data(train_df)
val_df = preprocess_data(val_df)
test_df = preprocess_data(test_df)

# Calculate the correlation matrix
correlation_matrix = train_df.corr()

# Identify the target variable correlation
target_correlation = correlation_matrix['Reviewer_Score']

# Sort the correlation values
sorted_correlation = target_correlation.abs().sort_values(ascending=False)

# Select the top features
top_features = sorted_correlation[1:6].index.tolist()

# top features = ['Review_Total_Negative_Word_Counts', 'Negative_Review', 'Average_Score',
# 'Review_Total_Positive_Word_Counts', 'Positive_Review']


# Update feature matrices
X_train = train_df[top_features].to_numpy()
X_val = val_df[top_features].to_numpy()
X_test = test_df[top_features].to_numpy()

y_train = train_df['Reviewer_Score'].to_numpy()
y_val = val_df['Reviewer_Score'].to_numpy()
y_test = test_df['Reviewer_Score'].to_numpy()

# LOGISTIC REGRESSION

# random_state =>  ensuring consistent results when the code is run multiple times.
clf = LogisticRegression(random_state=1).fit(X_train, y_train)

start_time = time.time()
# predicts the target variable
y_val_pred = clf.predict(X_val)
end_time = time.time()
total_training_time_lr = end_time - start_time

start_time = time.time()
y_test_pred = clf.predict(X_test)
end_time = time.time()
total_test_time_lr = end_time - start_time

val_accuracy_lr = accuracy_score(y_val, y_val_pred)
test_accuracy_lr = accuracy_score(y_test, y_test_pred)

print("Validation accuracy for Logistic Regression:", val_accuracy_lr)
print("Test accuracy for Logistic Regression:", test_accuracy_lr)
print("Total test time for Logistic Regression:", total_test_time_lr)
print("")
# Save Logistic Regression model
with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(clf, f)


# Decision Tree Classifier
DTC = DecisionTreeClassifier(random_state=1)
DTC.fit(X_train, y_train)

start_time = time.time()
y_val_pred_dt = DTC.predict(X_val)
end_time = time.time()
total_training_time_dt = end_time - start_time

start_time = time.time()
y_test_pred_dt = DTC.predict(X_test)
end_time = time.time()
total_test_time_dt = end_time - start_time

val_accuracy_dt = accuracy_score(y_val, y_val_pred_dt)
test_accuracy_dt = accuracy_score(y_test, y_test_pred_dt)

print("Validation accuracy for Decision Tree Classifier:", val_accuracy_dt)
print("Test accuracy for Decision Tree Classifier:", test_accuracy_dt)
print("Total test time for Decision Tree Classifier:", total_test_time_dt)
print("")
# Save Decision Tree Classifier model
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(DTC, f)

# Random Forest Classifier

RFC = RandomForestClassifier(n_estimators=150, max_depth=25, min_samples_leaf=75, oob_score=True, ccp_alpha=0.00001,
                             random_state=1)
RFC.fit(X_train, y_train)

start_time = time.time()
y_val_pred_rf = RFC.predict(X_val)
end_time = time.time()
total_training_time_rf = end_time - start_time

start_time = time.time()
y_test_pred_rf = RFC.predict(X_test)
end_time = time.time()
total_test_time_rf = end_time - start_time

val_accuracy_rf = accuracy_score(y_val, y_val_pred_rf)
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)

print("Validation accuracy for Random Forest Classifier:", val_accuracy_rf)
print("Test accuracy for Random Forest Classifier:", test_accuracy_rf)
print("Total test time for Random Forest Classifier:", total_test_time_rf)
print("")
# Save Random Forest Classifier model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(RFC, f)

# K-Nearest Neighbors Classifier
KNC = KNeighborsClassifier(n_neighbors=21)
KNC.fit(X_train, y_train)

start_time = time.time()
y_val_pred_kn = KNC.predict(X_val)
end_time = time.time()
total_training_time_kn = end_time - start_time

start_time = time.time()
y_test_pred_kn = KNC.predict(X_test)
end_time = time.time()
total_test_time_kn = end_time - start_time

val_accuracy_kn = accuracy_score(y_val, y_val_pred_kn)
test_accuracy_kn = accuracy_score(y_test, y_test_pred_kn)

print("Validation accuracy for K-Nearest Neighbors Classifier:", val_accuracy_kn)
print("Test accuracy for K-Nearest Neighbors Classifier:", test_accuracy_kn)
print("Total test time for K-Nearest Neighbors Classifier:", total_test_time_kn)
print("")
# Save K-Nearest Neighbors Classifier model
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(KNC, f)


# Visualize the results
labels = ['Logistic Regression', 'Random Forest Classifier', 'Decision Tree Classifier',
          'K-Nearest Neighbors Classifier']
val_accuracies = [val_accuracy_lr, val_accuracy_rf, val_accuracy_dt, val_accuracy_kn]
test_accuracies = [test_accuracy_lr, test_accuracy_rf, test_accuracy_dt, test_accuracy_kn]
training_times = [total_training_time_lr, total_training_time_rf, total_training_time_dt, total_training_time_kn]
test_times = [total_test_time_lr, total_test_time_rf, total_test_time_dt, total_test_time_kn]

# Classification accuracy bar graph
plt.figure(figsize=(10, 6))
plt.bar(labels, val_accuracies, color='blue', alpha=0.7, label='Validation Accuracy')
plt.bar(labels, test_accuracies, color='green', alpha=0.7, label='Test Accuracy')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy')
plt.legend()
plt.show()

# Total training time bar graph
plt.figure(figsize=(10, 6))
plt.bar(labels, training_times, color='orange', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Total Training Time')
plt.title('Total Training Time')
plt.show()

# Total test time bar graph
plt.figure(figsize=(10, 6))
plt.bar(labels, test_times, color='purple', alpha=0.7)
plt.xlabel('Models')
plt.ylabel('Total Test Time')
plt.title('Total Test Time')
plt.show()
