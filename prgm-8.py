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