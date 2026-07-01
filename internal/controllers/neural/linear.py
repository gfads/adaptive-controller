import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline


# Step 1: Load the dataset
print("Step 1: Load the dataset")
df = pd.read_csv( "data-training.csv", sep=';' )

# Step 2: Split features and target
print("Step 2: Split features and target")
X = df[['PC']].values  # Input feature
y = df['Rate'].values  # Target

# Step 3: Normalize the input features
print("Step 3: Normalize the input features")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split into train and test sets
print("Step 4: Split into train and test sets\n")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)


linear_models = ['lr', 'ridge', 'ridgeCV', 'SGDRegressor']
for lm in linear_models:
    print('Gerando resultados para o modelo ', lm)
    if lm == 'lr':
        model = LinearRegression().fit(X_train, y_train)
    elif lm =='ridge':
        model = Ridge(alpha=1.0).fit(X_train, y_train)
    elif lm=='ridgeCV':
        model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]).fit(X_train, y_train)
    else:
        model = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))
        model.fit(X_train, y_train)

    # Step 8: Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    nrmse = rmse / np.mean(y_test)
    print(f"NRMSE: {nrmse:.2f}")

    # Optional: Show predictions
    for i in range(len(y_test)):
        pc_value = scaler.inverse_transform([X_test[i]])[0][0]
        #print(f"PC={pc_value:.0f}, Actual Rate={y_test[i]}, Predicted={y_pred[i]:.2f}")
        print(f"{pc_value:.0f}; {y_pred[i]:.0f}")

    print('')

