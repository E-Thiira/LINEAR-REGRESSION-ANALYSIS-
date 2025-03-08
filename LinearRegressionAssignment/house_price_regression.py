import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('train.csv')

# Selecting relevant columns
df_selected = df[['GrLivArea', 'BedroomAbvGr', 'YearBuilt', 'YrSold', 'SalePrice']].copy()

# Compute house age
df_selected['Age'] = df_selected['YrSold'] - df_selected['YearBuilt']

# Drop unnecessary columns
df_selected = df_selected.drop(columns=['YearBuilt', 'YrSold'])

# Drop rows with missing values
df_selected = df_selected.dropna()

# Scatter plots for feature relationships with SalePrice
features = ['GrLivArea', 'BedroomAbvGr', 'Age']
plt.figure(figsize=(15, 5))

for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(x=df_selected[feature], y=df_selected['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('Sale Price')
    plt.title(f'{feature} vs Sale Price')

plt.tight_layout()
plt.show()

# Define features (X) and target variable (y)
X = df_selected[['GrLivArea', 'BedroomAbvGr', 'Age']]
y = df_selected['SalePrice']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R^2 Score: {r2:.4f}')

# Expected Output (example values, actual results may slightly vary)
# Mean Absolute Error (MAE): 31984.67
# Root Mean Squared Error (RMSE): 47390.77
# R^2 Score: 0.7072
