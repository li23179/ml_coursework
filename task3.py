import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# Load the data
file_name = "regression_insurance.csv"
data = pd.read_csv(file_name)

X = data.drop("charges", axis=1)
y = data["charges"]

# Define numerical and categorical features
numerical_features = ["age", "bmi", "children"]
catagorical_features = ["sex", "smoker", "region"]

# Split into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Transformer for numerical featuers
scaler = StandardScaler()

# One-Hot Encoder for categorical features
encoder = OneHotEncoder()

# Combine the preprocessing step
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", scaler, numerical_features),
        ("categorical", encoder, catagorical_features),
    ]
)

# fit the train data only into preprocessor
preprocessor.fit(X_train)

# apply both standard scaler and one hot encoder to the training and testing data
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

X_train_np = X_train_transformed.astype(np.float32)
X_test_np = X_test_transformed.astype(np.float32)

num_samples, num_features = X_train_np.shape

feature_names = preprocessor.get_feature_names_out()
feature_names = [name.split("__")[1] for name in feature_names]
print(feature_names)
model = pm.Model()

with model:
    # Define priors with y = Xw
    w0 = pm.Normal("w0", mu=0, sigma=20 )
    w = pm.Normal("w", mu=0, sigma=20, shape=num_features)
    sigma = pm.HalfNormal("sigma", sigma=10)

    # Linear Regression model : w0 + Xw
    y_est = w0 + pm.math.dot(X_train_np, w)

    likelihood = pm.Normal("y", mu=y_est, sigma=sigma, observed=y_train)

    # Inference - Hamiltonian MCMC with No U-Turn Sampler
    sampler = pm.NUTS()

    idata = pm.sample(num_samples, step=sampler, progressbar=True)

    # Summary data
    print(az.summary(idata, var_names=["w0", "w", "sigma"], round_to=3))

    # Posterior mean
    w0_mean = idata.posterior["w0"].mean().item()
    w_mean = idata.posterior["w"].mean(dim=("chain", "draw")).values
    sigma_mean = idata.posterior["sigma"].mean().item()

    print(f"Posterior mean intercept (w0): {w0_mean: .3f}")
    print(f"Posterior mean coefficients (w)")
    
    for name, coef in zip(feature_names, w_mean):
        print(f"{name:30s} = {coef: .3f}")
        
    print(f"Posterior mean noise std (sigma): {sigma_mean: .3f}")
    
    # Plot trace for all parameters
    az.plot_trace(idata, var_names=["w0", "w", "sigma"])
    plt.tight_layout()
    
    plt.savefig("regression_plots/traceplot.png", dpi=300, bbox_inches="tight")
    plt.show()