import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
file_name = "regression_insurance.csv"
data = pd.read_csv(file_name)

X = data.drop('charges', axis=1)
y = data['charges']

# Define numerical and categorical features
numerical_features = ['age', 'bmi', 'children']
catagorical_features = ['sex', 'smoker', 'region']

# Split into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformer for numerical featuers
scaler = StandardScaler()

# One-Hot Encoder for categorical features
encoder = OneHotEncoder()

# Combine the preprocessing step
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', scaler, numerical_features),
        ('categorical', encoder, catagorical_features),
    ]
)

# fit the train data only into preprocessor
preprocessor.fit(X_train)

# apply both standard scaler and one hot encoder to the training and testing data
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Train a linear regression model to predict the insurance charges.
lin_reg = LinearRegression()
lin_reg.fit(X_train_transformed, y_train)

# Print the learned coefficients for each feature: round each coefficient to
# three decimal places for clarity and clearly label each coefficient with its
# corresponding feature name.
feature_names = preprocessor.get_feature_names_out()
feature_names = [name.split("__")[1] for name in feature_names]

for name, coef in zip(feature_names, lin_reg.coef_):
    print(f"{name}: {coef: .3f}")

# Compute and print the Root Mean Squared
# Error (RMSE) on both training and test sets.
y_pred_train = lin_reg.predict(X_train_transformed)
y_pred_test = lin_reg.predict(X_test_transformed)

RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
print(f"Train RSME: {RMSE_train: .3f}")
RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"Test RSME: {RMSE_test: .3f}")

# R^2 scores
R2_train = r2_score(y_train, y_pred_train)
R2_test = r2_score(y_test, y_pred_test)

print(f"Train R^2: {R2_train:.3f}")
print(f"Test R^2: {R2_test:.3f}")

# Produce a scatter plot of
# predicted versus actual charges on the test set.
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(y_test, y_pred_test, edgecolors='black', linewidths=0.5)
ax.set_xlabel("Actual Charges")
ax.set_ylabel("Predict Charges")
ax.set_title("Linear Regression: Predicted vs Actual Charges (Test Set)")

# Regression line y = x for perfect prediction 
ax.plot( [y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'k--', linewidth=2, label="Perfect Prediction (y = x)")

ax.legend()

plt.savefig("regression_plots/linear_regression_plot.png", dpi=300)
plt.show()