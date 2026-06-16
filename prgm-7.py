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
importances = pd.Series(clf.feature_importances_, index=X.columns) 
importances.sort_values(ascending=False).plot(kind='bar', 
figsize=(8, 4)) 
plt.ylabel('Importance Score') 
plt.show() 