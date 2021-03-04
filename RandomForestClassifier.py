import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import train_test_split

data = pd.read_csv('F:\PSG Tech\Projects\PasswordStrength\cleaned.csv')

features = data.values[:, 0].astype('str')
labels = data.values[:, -1].astype('int')

""" 
vectorizer = TfidfVectorizer(min_df=0.0, analyzer='char')
features = vectorizer.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)
model = RandomForestClassifier(max_depth=50, criterion='entropy') 
"""

model = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=0.0, analyzer='char')),
    ('randfr', RandomForestClassifier(max_depth=50, criterion='entropy'))
])


model.fit(features, labels)

joblib.dump(model, "RandomForestClassifier.joblib")
#print('Training Accuracy: ',model.score(X_test, y_test))