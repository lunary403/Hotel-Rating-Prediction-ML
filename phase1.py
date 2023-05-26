import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Function to winsorize outliers in data
def winsorize_outliers(data, iqr_multiplier=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    upper_whisker = q3 + (iqr_multiplier * iqr)
    lower_whisker = q1 - (iqr_multiplier * iqr)
    data[data > upper_whisker] = upper_whisker
    data[data < lower_whisker] = lower_whisker
    return data


# Encoding function for positive reviews
def positiveRev_encoding(value):
    exclude_list = ['No Positive', 'Nothing']
    for exclude_string in exclude_list:
        if exclude_string in value:
            return 0
    return 1


# Encoding function for negative reviews
def negativeRev_encoding(value):
    exclude_list = ['No Negative', 'Nothing', 'Nothing', 'nothing', 'N A']
    for exclude_string in exclude_list:
        if exclude_string in value:
            return 0
    return 1


# Extract tags from tags string
def extract_tags(tags_str):
    tags = tags_str.strip("[]").replace("'", "").split(", ")
    tag_categories = {}
    for tag in tags:
        if "trip" in tag:
            tag_categories["trip_type"] = tag.strip()
        elif "Room" in tag:
            tag_categories["room_type"] = tag.strip()
        elif "Stayed" in tag:
            tag_categories["stay_duration"] = tag.strip()
        elif "Submitted" in tag:
            tag_categories["device_type"] = tag.strip()
        else:
            tag_categories["group_type"] = tag.strip()
    return tag_categories


# Extract country from address
def extract_country(address):
    address_parts = address.split()
    country = address_parts[-1]
    return country


# Preprocess the input DataFrame
def preprocess_data(df):
    # Winsorize outliers
    columns_to_winsorize = ['Additional_Number_of_Scoring', 'Average_Score', 'Review_Total_Negative_Word_Counts',
                            'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts',
                            'Total_Number_of_Reviews_Reviewer_Has_Given', 'lat', 'lng', 'Reviewer_Score']
    df[columns_to_winsorize] = df[columns_to_winsorize].apply(winsorize_outliers)

    # Fill missing values
    df['lat'].fillna(df['lat'].mode()[0], inplace=True)
    df['lng'].fillna(df['lng'].mode()[0], inplace=True)

    # Convert data types
    df['days_since_review'] = df['days_since_review'].str.replace("[days]", '').astype(int)
    df['Review_Date'] = pd.to_datetime(df['Review_Date'])

    # Extract date features
    df['month'] = df['Review_Date'].dt.month
    df['year'] = df['Review_Date'].dt.year
    df['day_of_week'] = df['Review_Date'].dt.day_name()
    df['day_of_week'] = df['day_of_week'].map({'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                                               'Friday': 4, 'Saturday': 5, 'Sunday': 6})

    # Drop unnecessary columns
    df = df.drop(['Review_Date'], axis=1)

    # Apply text encoding and cleaning
    df['Positive_Review'] = df['Positive_Review'].apply(positiveRev_encoding)
    df['Negative_Review'] = df['Negative_Review'].apply(negativeRev_encoding)
    df['Hotel_Name'] = df['Hotel_Name'].str.replace('Hôtel', 'Hotel')  # Fix typo in hotel names

    # Extract tags from 'Tags' column
    tags_df = df['Tags'].apply(lambda x: pd.Series(extract_tags(x)))
    df = pd.concat([df, tags_df], axis=1)
    df.drop('Tags', axis=1, inplace=True)

    # Fill missing values and handle categorical variables
    df['trip_type'] = df['trip_type'].fillna('Leisure trip')
    df['trip_type'] = df['trip_type'].map({'Leisure trip': 1, 'Business trip': 0})
    df['room_type'] = df['room_type'].fillna('Double Room')
    df['device_type'] = df['device_type'].fillna('Submitted from a mobile device')
    df['Hotel_Address'] = df['Hotel_Address'].apply(extract_country)
    df['Hotel_Country'] = df['Hotel_Address']
    df['Reviewer_Nationality'] = df['Reviewer_Nationality'].str.split().str[-1]
    df['Similar_country'] = (df['Hotel_Address'] == df['Reviewer_Nationality']).astype(int)

    # Clean and convert 'stay_duration' column
    df['stay_duration'] = df['stay_duration'].str.replace("[Stayed, night,s]", '')
    mode_duration = df['stay_duration'].mode()[0]
    df['stay_duration'] = df['stay_duration'].fillna(mode_duration).astype(int)
    df['stay_duration'] = winsorize_outliers(df['stay_duration'])

    return df


# Read the CSV file and shuffle the DataFrame
df = pd.read_csv("hotel-regression-dataset.csv")
df_shuffled = df.sample(n=len(df), random_state=1)

# Preprocess the shuffled DataFrame
df_shuffled = preprocess_data(df_shuffled)

# Select relevant columns for the final DataFrame
df_final = df_shuffled[
    ['Additional_Number_of_Scoring', 'Average_Score', 'Negative_Review', 'Review_Total_Negative_Word_Counts',
     'Total_Number_of_Reviews', 'Positive_Review', 'Review_Total_Positive_Word_Counts',
     'Total_Number_of_Reviews_Reviewer_Has_Given', 'days_since_review', 'lat', 'lng', 'month', 'year', 'day_of_week',
     'trip_type', 'stay_duration', 'Similar_country', 'Reviewer_Score']]

# Split the final DataFrame into train, validation, and test sets
train_df, val_df, test_df = df_final[:174189], df_final[174189:232252], df_final[232252:]

# Split the features and target variables
X_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
X_val, y_val = val_df.iloc[:, :-1].values, val_df.iloc[:, -1].values
X_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values

# Scale the numerical features using MinMaxScaler
scaler = MinMaxScaler().fit(X_train[:, :8])


def preprocessor(X):
    A = np.copy(X)
    A[:, :8] = scaler.transform(A[:, :8])
    return A


# Preprocess the features
X_train, X_val, X_test = preprocessor(X_train), preprocessor(X_val), preprocessor(X_test)

# Train and evaluate Linear Regression model
lm = LinearRegression().fit(X_train, y_train)
train_mse_lm = mse(lm.predict(X_train), y_train)
val_mse_lm = mse(lm.predict(X_val), y_val)

# Train and evaluate K-Nearest Neighbors model
knn = KNeighborsRegressor(n_neighbors=14).fit(X_train, y_train)
train_mse_knn = mse(knn.predict(X_train), y_train)
val_mse_knn = mse(knn.predict(X_val), y_val)

# Train and evaluate Random Forest model
rfr = RandomForestRegressor(max_depth=10).fit(X_train, y_train)
train_mse_rfr = mse(rfr.predict(X_train), y_train)
val_mse_rfr = mse(rfr.predict(X_val), y_val)

# Train and evaluate Gradient Boosting model
gbr = GradientBoostingRegressor(n_estimators=100).fit(X_train, y_train)
train_mse_gbr = mse(gbr.predict(X_train), y_train)
val_mse_gbr = mse(gbr.predict(X_val), y_val)

# Calculate MSE for the Gradient Boosting model predictions on the test set
gbr_pred = gbr.predict(X_test)
test_mse_gbr = mse(gbr_pred, y_test)

# Calculate the score for the Gradient Boosting model on the test set
score_gbr = gbr.score(X_test, y_test)

print("Train MSE (Linear Regression):", train_mse_lm)
print("Validation MSE (Linear Regression):", val_mse_lm)
print("Train MSE (K-Nearest Neighbors):", train_mse_knn)
print("Validation MSE (K-Nearest Neighbors):", val_mse_knn)
print("Train MSE (Random Forest):", train_mse_rfr)
print("Validation MSE (Random Forest):", val_mse_rfr)
print("Train MSE (Gradient Boosting):", train_mse_gbr)
print("Validation MSE (Gradient Boosting):", val_mse_gbr)
print("Test MSE (Gradient Boosting):", test_mse_gbr)
print("Gradient Boosting Score:", score_gbr)
