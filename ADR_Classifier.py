#importing the Necessary Libraries
import pandas as pd

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

#Importing the Dataset

df=pd.read_csv("Data.csv")
df.head()

#Converting Into Lower Case
df['low_tweet']=df['Tweet'].values
df['low_tweet']=df['low_tweet'].apply(lambda x: x.lower())
df['low_tweet'].head()

#Tokenizing the Tweets
tokenizer = RegexpTokenizer(r'\w+')
df['token_tweet'] = df['low_tweet'].apply(lambda row: tokenizer.tokenize(row))
df['token_tweet'].head()


#Filter out Stop Words
nltk.download('stopwords')
no_use_words=set(stopwords.words('english'))
df['stop_filter_tweet']=df['token_tweet'].apply(lambda row: [x for x in row if not x in no_use_words])
df['stop_filter_tweet'].head()

#lemmatizing the Tweets
nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()
df['Lemmat_tweets']=df['stop_filter_tweet'].apply(lambda x:[lemmatizer.lemmatize(a) for a in x])
df['Lemmat_tweets'].head()

#Splitting into Training and Test set
data=df.copy()
sentences = data['Lemmat_tweets'] 
y = data['ADR_label']
sentences=sentences.astype(str)
train_x,test_x,train_y,test_y = train_test_split(sentences, y, test_size=0.3, random_state=500)

#Applying Bag of words
vectorizer = CountVectorizer()
vectorizer.fit(train_x)
X_train = vectorizer.transform(train_x)
X_test  = vectorizer.transform(test_x)
X_train

#Applying Logistic Regression
model_LR = LogisticRegression(class_weight='balanced')
model_LR.fit(X_train,train_y)
score_LR = model_LR.score(X_test,test_y)
score_LR

pred=model_LR.predict(X_test)
cm_LR = confusion_matrix(test_y, pred)
print(cm_LR)

#Applying Random Forest
model_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
model_RF.fit(X_train, train_y)
y_pred = model_RF.predict(X_test)
score_RF = model_RF.score(X_test,test_y)
score_RF

# Making the Confusion Matrix
cm_RF = confusion_matrix(test_y, y_pred)
print(cm_RF)

