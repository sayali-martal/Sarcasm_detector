# Natural Language Processing

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_json('Sarcasm_Headlines_Dataset.json', lines = True)
dataset = dataset.drop(['article_link'], axis = 1)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 26709):
    headline = re.sub('[^a-zA-Z]', ' ', dataset['headline'][i])
    headline = headline.lower()
    headline = headline.split()
    ps = PorterStemmer()
    headline = [ps.stem(word) for word in headline if not word in set(stopwords.words('english'))]
    headline = ' '.join(headline)
    corpus.append(headline)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = None)
#X = cv.fit_transform(corpus).toarray()
X = cv.fit_transform(corpus)
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Calculate accuracy_score
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)