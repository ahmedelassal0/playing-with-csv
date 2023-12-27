# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, mean_squared_log_error, median_absolute_error, r2_score

# Importing the 'clean_df' DataFrame from the 'config' module
from config import clean_df

# Extract relevant columns from the DataFrame
data = clean_df.loc[
    :,
    ('price', 'name', 'host_id', 'room_type',
     'minimum_nights', 'number_of_reviews',
     'reviews_per_month',
     'availability_365'
     )
]

# Create a DataFrame using the extracted data
df = pd.DataFrame(data)

# Remove the 'name' column from features
# 'name' column is removed as it's unlikely to contribute to numerical predictions
x = df[['host_id', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']]

# Target variable (what we want to predict)
y = df['price']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create a base decision tree regressor
base_regressor = DecisionTreeRegressor(random_state=42)

# Create a bagging regressor using the base regressor
bagging_regressor = BaggingRegressor(base_regressor, n_estimators=10, random_state=42)

# Train the bagging regressor on the training data
bagging_regressor.fit(x_train, y_train)

y_pred = bagging_regressor.predict(x_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
msle = mean_squared_log_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Explained Variance Score (EVS): {evs}')
print(f'Mean Squared Log Error (MSLE): {msle}')
print(f'Median Absolute Error: {medae}')
print(f'R-squared (R2): {r2}')
# Make predictions on the test set
# Evaluate the performance of the bagging regressor using various metrics
# Mean Absolute Error (MAE): Represents the average absolute difference between predicted and actual values
# Mean Squared Error (MSE): Represents the average squared difference between predicted and actual values
# Explained Variance Score (EVS): Measures the proportion of variance in the target variable that is explained by the model
# Mean Squared Log Error (MSLE): Measures the mean of the logarithmic squared differences between predicted and actual values
# Median Absolute Error: Represents the median absolute difference between predicted and actual values
# R-squared (R2): Represents the proportion of variance in the dependent variable that is predictable from the independent variables
