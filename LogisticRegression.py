import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
#from sklearn.model_selection import train_test_split

data = pd.read_csv('F:\PSG Tech\Projects\PasswordStrength\cleaned.csv')

features = data.values[:, 0].astype('str')
labels = data.values[:, -1].astype('int')

#vectorizer = TfidfVectorizer(min_df=0.0, analyzer='char')
#features = vectorizer.fit_transform(features)


""" print(vectorizer.get_feature_names())
print(vectorizer.vocabulary_)
m = features.todense()
print(m)
 """

#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)
#model = LogisticRegression(multi_class='multinomial', solver='sag')

model = Pipeline([
    ('tfidf', TfidfVectorizer(min_df=0.0, analyzer='char')),
    ('logreg', LogisticRegression(multi_class='ovr', solver='sag'))
])

model.fit(features, labels)

joblib.dump(model, 'LogisticRegression.joblib')

#print('Training Accuracy: ',model.score(X_test, y_test))