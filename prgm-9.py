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