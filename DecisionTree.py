import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

data = pd.read_csv('F:\PSG Tech\Projects\PasswordStrength\cleaned.csv')

features = data.values[:, 0].astype('str')
labels = data.values[:, -1].astype('int')

model=Pipeline([('tfidf',TfidfVectorizer(analyzer="char")),('decisionTree',DecisionTreeClassifier())])
model.fit(features,labels)
joblib.dump(model,'DecisionTree.joblib')
