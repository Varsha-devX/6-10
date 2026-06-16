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


prgm-7
code:
import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, accuracy_score 
import matplotlib.pyplot as plt 
import seaborn as sns 
# 1. Simulate Churn Dataset 
np.random.seed(42) 
n = 1000 
data = pd.DataFrame({ 
'Tenure': np.random.randint(1, 72, n),  # months 
'MonthlyCharges': np.random.uniform(20, 120, n), 
'TotalCharges': lambda df: df['Tenure'] * df['MonthlyCharges'], 
'SupportCalls': np.random.poisson(2, n), 
'StreamingService': np.random.choice([0, 1], size=n), 
'PaperlessBilling': np.random.choice([0, 1], size=n), 
'Contract': np.random.choice([0, 1, 2], size=n),  # 0=Month-to-month, 1=One year, 2=Two 
year 
}) 
data['TotalCharges'] = data['Tenure'] * data['MonthlyCharges'] 
# Simulate churn label (higher risk for short tenure + high support calls) 
data['Churn'] = ((data['Tenure'] < 12) & (data['SupportCalls'] > 3)).astype(int) 
# 2. Features and Labels 
X = data.drop('Churn', axis=1) 
y = data['Churn'] 
# 3. Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# 4. Train Decision Tree 
clf = DecisionTreeClassifier(max_depth=4, random_state=42) 
clf.fit(X_train, y_train) 
# 5. Evaluate 
y_pred = clf.predict(X_test) 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("\nClassification Report:\n", classification_report(y_test, y_pred)) 
# 6. Visualize the Tree 
plt.figure(figsize=(18, 10)) 
plot_tree(clf, feature_names=X.columns, class_names=['No Churn', 'Churn'], filled=True) 
plt.title("Decision Tree for Customer Churn") 
plt.show() 
# 7. Feature Importance 
pg. 18 
Dept. of CSE – AI & ML 
importances = pd.Series(clf.feature_importances_, index=X.columns) 
importances.sort_values(ascending=False).plot(kind='bar', 
figsize=(8, 4)) 
plt.ylabel('Importance Score') 
plt.show() 

prgm-8
code:
import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import seaborn as sns 
import matplotlib.pyplot as plt 
# 1. Load dataset 
iris = load_iris() 
X = pd.DataFrame(iris.data, columns=iris.feature_names) 
y = pd.Series(iris.target, name='Target') 
# 2. Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# 3. Random Forest with default parameters 
rf = RandomForestClassifier(random_state=42) 
rf.fit(X_train, y_train) 
y_pred = rf.predict(X_test) 
# 4. Evaluate model 
print("Random Forest Classifier (Default Parameters):") 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("Classification Report:\n", classification_report(y_test, y_pred)) 
# 5. Confusion Matrix 
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', 
xticklabels=iris.target_names, yticklabels=iris.target_names, cmap="Blues") 
plt.title("Confusion Matrix") 
plt.show() 
# 6. Hyperparameter Tuning: GridSearch 
param_grid = { 
'n_estimators': [10, 50, 100], 
'max_depth': [None, 3, 5, 10], 
'min_samples_split': [2, 4], 
} 
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5) 
pg. 21 
grid.fit(X_train, y_train) 
Dept. of CSE – AI & ML 
print("Best Parameters:", grid.best_params_) 
# 7. Evaluate Best Model 
best_rf = grid.best_estimator_ 
y_pred_best = best_rf.predict(X_test) 
print("\nRandom Forest Classifier (Tuned Parameters):") 
print("Accuracy:", accuracy_score(y_test, y_pred_best)) 
print("Classification Report:\n", classification_report(y_test, y_pred_best)) 
# 8. Feature Importance Plot 
importances = pd.Series(best_rf.feature_importances_, index=X.columns) 
importances.sort_values().plot(kind='barh', title='Feature Importances') 
plt.xlabel("Importance Score") 
plt.show() 

prgm-9
code:
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import accuracy_score, classification_report 
# 1. Load dataset (after downloading and placing 'spam.csv' in your folder) 
df 
= pd.read_csv("/content/drive/MyDrive/Colab 
encoding='latin-1')[['v1', 'v2']] 
df.columns = ['label', 'message'] 
# 2. Encode labels 
df['label'] = df['label'].map({'ham': 0, 'spam': 1}) 
# 3. Text vectorization 
vectorizer = CountVectorizer() 
X = vectorizer.fit_transform(df['message']) 
y = df['label'] 
Notebooks/Datasets/spam.csv", 
# 4. Train/test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# 5. Train Naive Bayes model 
model = MultinomialNB() 
model.fit(X_train, y_train) 
# 6. Predict & evaluate 
y_pred = model.predict(X_test) 
print("Accuracy:", accuracy_score(y_test, y_pred)) 
print("\nClassification Report:\n", classification_report(y_test, y_pred))

prgm-10
code:
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
# 1. Simulate customer RFM data 
np.random.seed(42) 
n_customers = 200 
data = pd.DataFrame({ 
'CustomerID': range(1, n_customers + 1), 
'Recency': np.random.randint(1, 100, size=n_customers),     
# days since last purchase 
'Frequency': np.random.randint(1, 20, size=n_customers),    # number of purchases 
'Monetary': np.random.randint(100, 10000, size=n_customers) # total money spent 
}) 
# 2. Standardize the RFM features 
scaler = StandardScaler() 
rfm_scaled = scaler.fit_transform(data[['Recency', 'Frequency', 'Monetary']]) 
# 3. Find optimal number of clusters using the Elbow Method 
sse = [] 
for k in range(1, 11): 
kmeans = KMeans(n_clusters=k, random_state=42) 
kmeans.fit(rfm_scaled) 
sse.append(kmeans.inertia_) 
# Plot Elbow Curve 
plt.figure(figsize=(6, 4)) 
plt.plot(range(1, 11), sse, marker='o') 
plt.title("Elbow Method for Optimal K") 
plt.xlabel("Number of clusters") 
plt.ylabel("SSE (Inertia)") 
plt.grid(True) 
plt.show() 
# 4. Apply K-Means with optimal clusters (e.g., k=4) 
kmeans = KMeans(n_clusters=4, random_state=42) 
data['Cluster'] = kmeans.fit_predict(rfm_scaled) 
# 5. Visualize Clusters 
plt.figure(figsize=(8, 6)) 
sns.scatterplot(data=data, x='Recency', y='Monetary', hue='Cluster', palette='Set2') 
pg. 26 
Dept. of CSE – AI & ML 
plt.title("Customer Segmentation (Recency vs Monetary)") 
plt.show() 
plt.figure(figsize=(8, 6)) 
sns.scatterplot(data=data, x='Frequency', y='Monetary', hue='Cluster', palette='Set1') 
plt.title("Customer Segmentation (Frequency vs Monetary)") 
plt.show()