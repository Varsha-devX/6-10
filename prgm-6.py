prgm-6
code:
from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.preprocessing import PolynomialFeatures, StandardScaler 
from sklearn.metrics import mean_squared_error, r2_score 
# 1. Load the dataset 
boston = fetch_california_housing() 
X = boston.data 
y = boston.target 
# 2. Split into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
# 3. Standardize the features 
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) 
# 4. Linear Regression 
lr = LinearRegression() 
lr.fit(X_train_scaled, y_train) 
y_pred_lr = lr.predict(X_test_scaled) 
# 5. Polynomial Regression (degree 2) 
poly = PolynomialFeatures(degree=2) 
X_train_poly = poly.fit_transform(X_train_scaled) 
X_test_poly = poly.transform(X_test_scaled) 
pr = LinearRegression() 
pr.fit(X_train_poly, y_train) 
y_pred_pr = pr.predict(X_test_poly) 
# 6. Ridge Regression 
ridge = Ridge(alpha=1.0) 
ridge.fit(X_train_scaled, y_train) 
y_pred_ridge = ridge.predict(X_test_scaled) 
# 7. Lasso Regression 
lasso = Lasso(alpha=0.1) 
lasso.fit(X_train_scaled, y_train) 
y_pred_lasso = lasso.predict(X_test_scaled) 
# 8. Evaluation function 
def evaluate(name, y_true, y_pred): 
print(f"{name}:") 
print(f"  MSE: {mean_squared_error(y_true, y_pred):.2f}") 
print(f"  R2 : {r2_score(y_true, y_pred):.2f}") 
print("-" * 30) 
# 9. Compare models 
evaluate("Linear Regression", y_test, y_pred_lr) 
evaluate("Polynomial Regression", y_test, y_pred_pr) 
evaluate("Ridge Regression", y_test, y_pred_ridge) 
evaluate("Lasso Regression", y_test, y_pred_lasso)