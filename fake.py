import imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import json
import re
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn import datasets
from textblob import TextBlob
from wordcloud import STOPWORDS, WordCloud
from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


with open('reviews.json') as project_file:    
    data = json.load(project_file)
dataset=pd.json_normalize(data) 
print(dataset.head()) 

 
print(STOPWORDS)

def clean_text(text):
    #Remove RT
    text = re.sub(r'RT', '', text)
    
    #Fix &
    text = re.sub(r'&amp;', '&', text)
    
    #Remove punctuations
    text = re.sub(r"[?!.;:,'#@-]", '', text)

    #removes unicodes(symbols that seem weird)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    print(text)
    #filter to allow only alphabets
    text=re.sub(r'[^a-zA-Z\']', ' ', text)
    print(text)
    #Convert to lowercase to maintain consistency
    text = text.lower()
    return text

dataset['clean'] = dataset['text'].apply(clean_text)
print(dataset['clean'])

#tokenization
def gettokens(text):
    token_space = tokenize.WhitespaceTokenizer()
    all_words = ' '.join([text for text in dataset['text']])
    token_phrase = token_space.tokenize(all_words)
    return token_phrase

dataset['newphrase']=dataset['text'].apply(gettokens) 
print(dataset['newphrase'])
 
'''
#stemming 
dataset['clean_text'] = dataset.text.apply(lambda x: clean_text(x))
st=PorterStemmer()
dataset['text'] = dataset['text'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
print(dataset)
'''


#generate word frequency
def gen_freq(text):
    word_list=[] #stores the list of words
        
    for words in text.split(): #Loop over all the reviews and extract words into word_list
        word_list.extend(words)

    word_freq=pd.Series(word_list).value_counts() #Create word frequencies using word_list

    word_freq[:20]

     #Print top 20 word
    print(word_freq)
    return word_freq[:20]
      
gen_freq(dataset.text.str)

#removing stopwords
text = dataset.text.apply(lambda x: clean_text(x))
word_freq = gen_freq(text.str)*100
word_freq = word_freq.drop(labels=STOPWORDS, errors='ignore')
print(word_freq)
print(text)

#gets all the adjectives
def getAdjectives(text):

    blob=TextBlob(text)
    return [ word for (word,tag) in blob.tags if tag == "JJ"]

dataset['adjectives'] = dataset['text'].apply(getAdjectives)
#print(dataset)
adjectives=dataset['adjectives']
print(adjectives)

#obtain adjectives and their polarity generally
dataset[['polarity', 'subjectivity']] = dataset['text'].apply(lambda text: pd.Series(TextBlob(text).sentiment))
dataset = dataset.explode("adjectives") #obtain adjectives and their polarity individually 
dataset= dataset.dropna() #drop NaN values
print(dataset)
print(dataset[['adjectives', 'polarity']])

positive_list=[]
neutral_list=[]
negative_list=[]

#generating list of positive words
print('Positive:')
positive_list= dataset.loc[dataset['polarity'] > 0.2, ['adjectives']]
print(positive_list)
#generating frequency of the positive words
pos_freq=positive_list.value_counts()
#word_freq=pos_freq
print(pos_freq)
#generate wordcloud of positive words
wc = WordCloud(width= 800, height= 500, max_font_size = 110, collocations = False).generate(str(positive_list))

plt.figure(figsize=(10,7))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

#generating list of neutral words
print('Neutral:')
neutral_list=dataset.loc[dataset['polarity'] == 0, ['adjectives']]
print(neutral_list)
#generating frequency of the neutral words
neu_freq=neutral_list.value_counts()
print(neu_freq)
#generate wordcloud of neutral words
wc = WordCloud(width= 800, height= 500, max_font_size = 110, collocations = False).generate(str(neutral_list))

plt.figure(figsize=(10,7))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

#generating list of negative words        
print('Negative:')
negative_list=dataset.loc[dataset['polarity'] < 0.3, ['adjectives']]
print(negative_list)
#generating frequency of the negative words
neg_freq=negative_list.value_counts()
print(neg_freq)
#generate wordcloud of negative words
wc = WordCloud(width= 800, height= 500, max_font_size = 110, collocations = False).generate(str(negative_list))

plt.figure(figsize=(10,7))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()


#creating a model
from sklearn import metrics
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Split the data
X_train,X_test,y_train,y_test = train_test_split(dataset['text'], dataset['adjectives'], test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# Vectorizing and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])
                                           
# Fitting the model
model = pipe.fit(X_train, y_train)

# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))

cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=['positive', 'neutral', 'negative'])



