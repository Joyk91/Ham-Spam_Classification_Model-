# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 04:08:51 2019

@author: joyk9
"""

import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re
import string
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import classification_report,confusion_matrix  
from sklearn.pipeline import Pipeline 
import nltk 
import pickle
nltk.download('wordnet')
import email
from nltk.stem.snowball import SnowballStemmer 
from sklearn.metrics import classification_report,confusion_matrix 

#################################################################################################################
""" part 1 Loading the data  #####################################################################""" 
##################################################################################################################

""" have to end path with \\* and windows have to have double \\ or wont work """
# load in ham dataset 
path_ham = "C:\\Users\\joyk9\\Documents\\machine learning\\assignment 1\\project emails\\emails for project\\ham emails\\*txt"
ham_files = glob.glob(path_ham)
list_ham = []
print(list_ham)
for file_path in ham_files:
   with open(file_path) as f:
        list_ham.append(f.read())

# Give specific dummy variables to ham eamils 
ham = pd.DataFrame(list_ham, columns = ["emails"])
ham["target"]=0
print(ham.head(10))

# load in spam dataset 
path_spam = "C:\\Users\\joyk9\\Documents\\machine learning\\assignment 1\\project emails\\emails for project\\spam emails\\BG\\BG\\2004\\08\\*.txt"
spam_files = glob.glob(path_spam)
list_spam = []


for file_path in spam_files:
    with open(file_path) as f:
        list_spam.append(f.read())
print(list_spam) 

# give specific dummt variables to spam emails 
spam = pd.DataFrame(list_spam, columns = ["emails"])
spam["target"]=1


allEmails = pd.concat([ham, spam])  # merge both datasets into one 
allEmails = allEmails.sample(frac=1).reset_index(drop=True) # mix the dataset  

print(allEmails.info()) # let's get some infor on the dataset 
allEmails.groupby('target').describe()  # Let's see how many ham and spam emails are in our dataset 


""" -------------------------------------------------------------------------------------------------"""
""" **************************** PRE PROCESSING ***************************************************"""
""" -----------------------------------------------------------------------------------------------"""

## pre-processing Helper functions 
# will separate the email into it's different parts 
def extract_text(email):
    '''need to map all parts of the emails so as to get the body of text'''
    parts = []
    for part in email.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts) 


# This function will strip the email addresses fro the emails
def split_email_addresses(line):
    '''Seperate email addresses'''
    if line:
        addrs = line.split(',')
        addrs = frozenset(map(lambda x: x.strip(), addrs))
    else:
        addrs = None
    return addrs
        
# now that the to functions above are working have to parse the emails into a list of emails objects
messages = list(map(email.message_from_string, allEmails['emails']))
allEmails.drop('emails', axis=1, inplace=True) 


# return the fields of each email object in list
keys = messages[0].keys()
for key in keys:
    allEmails[key] = [f[key] for f in messages] 
    
    
# Parse content from emails
allEmails['content'] = list(map(extract_text, messages)) 


# Split multiple email addresses
allEmails['From'] = allEmails['From'].map(split_email_addresses)
allEmails['To'] = allEmails['To'].map(split_email_addresses)

print(allEmails)
del messages

allEmails.head() 
allEmails.info()


# create a new data-frame and fill it with the columns we want == subject, target, content 
email_df = pd.DataFrame()
email_df['subject'] = allEmails['Subject']
email_df['emails'] = allEmails['content']
email_df['target'] = allEmails['target'] 

""" do not want to get rid of subject because i think that is very important in classifying 
spam and ham. however will try and remove the wors sunject. so have to merge the cols subject 
and emails back together and then filter for reg expression etc """ 



email_df["emails"] = email_df["subject"].map(str) + email_df["emails"]
print(email_df.head(5)) # our dataset is looking very well now


""" --------------------------------------------------------------------------------------------------------------------------""" 
"""******************************************* Train and Test Split *********************************************************""" 
"""------------------------------------------------------------------------------------------------------------------------------"""

X_train, X_test, y_train, y_test = train_test_split(email_df["emails"], email_df["target"], test_size=0.3, random_state = 11)

train_set = pd.concat([X_train, y_train], axis=1, join_axes=[X_train.index])

train_set.to_csv("train_data.csv", sep="\t", encoding = "utf-8", index=False)

test_set = pd.concat([X_test, y_test], axis=1, join_axes=[X_test.index])

test_set.to_csv("test_data.csv", sep="\t", encoding = "utf-8", index=False)

print(train_set) 
len(train_set)  # 2272 emails 


# lets get some stats on the data 
train_set.groupby('target').describe()
# as we can see by this command the train set it made up of 1299 non-spam eamils and 973 spam emails 
#therefore its easy to conclude that this dataset is slightly imbalanced  
# can also see that 1245 of the non-spam emails are unique and 866 of spam are unigue 


"""----------------------------------------------------------------------------------------------------------------------------""" 
"""******************************************************* Feature Extraction ************************************************""" 
"""-----------------------------------------------------------------------------------------------------------------------------"""
"""Stemming, removing regular expression, remove punctuation and filtering of stop words are included in this 
classification pipeline (see Fig 4). As domain knowledge increased whilst working on the pipeline these feature 
extraction steps were continuously reviewed and updated.""" 

# we have to clean the data in order to get it compatible with ML algorithms 
# we have to remove a host of stopwords to make a truly robust model
def clean_text(text):
    stop = set(stopwords.words('english'))
    stop.update(("to","cc","subject","http","re", "ect", "html","lon", "hou", "etc", "www", "com", "from","sent","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
    exclude = set(string.punctuation) 
    #porter= SnowballStemmer("english")
    
    text=text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    stop_free = " ".join([i for i in text.lower().split() if((i not in stop) and (not i.isdigit()))])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    #stem = " ".join(porter.stem(token) for token in punc_free.split())
    
    return punc_free


# now lets test our cleaning function by looking at a before and sfter version of an email. 
train_set['emails'][3] # before clean_text function has been applied. Notice upper/lower case lettering, extra pinctuation stopwords etc
train_set['emails']= train_set['emails'].apply(clean_text)
train_set['emails'][3] # WOW that's alot cleaner than before. Our ML algorithms are going to like this much better
 

#########################################################################################################################################
""" EDA ###############################################################################################"""
#########################################################################################################################################
""" Now I want to get a better feel for the data. I need to have more subject matter on the data and I need to know what exactly
i am feeding my classification models. Only then can I truly interpret model results"""

category_count = pd.DataFrame()
category_count['count'] = train_set['target'].value_counts()
print(category_count)


######## PLot 1 barplot of spam and ham counts 

fig, ax = plt.subplots(figsize = (12, 6))
sns.barplot(x = category_count.index, y = category_count['count'], ax = ax)
ax.set_ylabel('Count', fontsize = 15)
ax.set_xlabel('Target',fontsize = 15)
ax.tick_params(labelsize=15) 
ax.set_title('Target Counts', fontsize = 15)
"""
the plot determines this is an imbalanced dataset. 
There are more ham texts than spam texts here. 
The typical problem with an imbalanced dataset is that the simple metric like accuracy or precision may not reflect 
the real performance of predictive models. 
"""

########## PLOT 2 top 20 words used in sapm and ham 
# first have to tokenize 
# then have to count 
# create new dataframe for plotting  
# then plot  
spam_df = train_set[train_set['target'] == 1] #create sub-dataframe of spam text
ham_df = train_set[train_set['target'] == 0] #sub-dataframe of ham text

def tokenize(text):
    tokens = nltk.word_tokenize(text) # tokenize the text
    tokens = [w for w in tokens if len(w) >=3] 
    return tokens


spam_df['tokens'] = spam_df['emails'].map(tokenize)
ham_df['tokens'] = ham_df['emails'].map(tokenize) 

""" top 20 words in spam and ham emails """
spam_words = []
for token in spam_df['tokens']:
    spam_words = spam_words + token #combine text in different columns in one list

ham_words = []
for token in ham_df['tokens']:
    ham_words += token

from collections import Counter
spam_count = Counter(spam_words).most_common(20)
ham_count = Counter(ham_words).most_common(20)

spam_count_df = pd.DataFrame(spam_count, columns = ['word', 'count'])
ham_count_df = pd.DataFrame(ham_count, columns = ['word', 'count'])
print(spam_count_df)
print(ham_count_df)

# top words in spam emails 
spam_count
fig, (ax,ax1) = plt.subplots(1,2,figsize = (18, 6))
sns.barplot(x = spam_count_df['word'], y = spam_count_df['count'], ax = ax)
ax.set_ylabel('count', fontsize = 15)
ax.set_xlabel('word',fontsize = 15)
ax.tick_params(labelsize=15) 
ax.set_xticklabels(spam_count_df['word'], rotation = 60)
ax.set_title('Spam top 20 words', fontsize = 15)

# top word in ham emails 
sns.barplot(x = ham_count_df['word'], y = ham_count_df['count'], ax = ax1)
ax1.set_ylabel('count', fontsize = 15)
ax1.set_xlabel('word',fontsize = 15)
ax1.tick_params(labelsize=15) 
ax1.set_xticklabels(ham_count_df['word'], rotation = 60)
ax1.set_title('Non-Spam top 20 words', fontsize = 15) 



################## PLOT 3 WordCloud  
# just for nice visual of top words in both spam and ham emails  
#utilise wordcloud 

spam_words_str = ' '.join(spam_words)
ham_words_str = ' '.join(ham_words)

spam_word_cloud = WordCloud(width = 600, height = 400, background_color = 'black').generate(spam_words_str)
ham_word_cloud = WordCloud(width = 600, height = 400,background_color = 'black').generate(ham_words_str)

fig, (ax, ax2) = plt.subplots(1,2, figsize = (18,8))
ax.imshow(spam_word_cloud)
ax.axis('off')
ax.set_title('spam word cloud', fontsize = 20)
ax2.imshow(ham_word_cloud)
ax2.axis('off')
ax2.set_title('ham word cloud', fontsize = 20)
plt.show()





 ##### length of spam and ham eamis 

spam_df['len'] = spam_df['emails'].apply(lambda x: len([w for w in x.split(' ')]))
ham_df['len'] = ham_df['emails'].apply(lambda x: len([w for w in x.split(' ')]))


# stats of spam length + processed length  
# stats lof ham lemth + processed length 
print ('spam length info')
print (spam_df[['len']].describe())
print ('ham length info')
print (ham_df[['len']].describe())

# histograms comparing lengths of spam and ham emails also comparing actuall length of email vs processed lenght 
fig, (ax) = plt.subplots(1,1,figsize = (18, 6)) 

spam_df['len'].plot.hist(bins = 20, ax=ax, edgecolor = 'white', color = 'orange')  

ax.tick_params(labelsize = 15)
ax.set_xlabel('length of sentence', fontsize = 12)
ax.set_ylabel('spam_frequency', fontsize = 12) 
ax.set_title('Length of Spam Emails')
ax.set_xlim([0,1000])


fig, (ax2) = plt.subplots(1,1,figsize = (18, 6)) 
ham_df['len'].plot.hist(bins = 20, ax = ax2, edgecolor = 'white', color = 'blue')


ax2.tick_params(labelsize = 15)
ax2.set_xlabel('length of sentence', fontsize = 12)
ax2.set_ylabel('ham_frequency', fontsize = 12) 
ax2.set_title('Length of Ham Emails')
ax2.set_xlim([0,1000])



""" plots to show the difference in length between spam and ham emails 
showed both processed and unprocessed in histograms for comparison""" 

"""----------------------------------------------------------------------------------------------------------------------------""" 
"""******************************************************* Feature Extraction Continued ************************************************""" 
"""-----------------------------------------------------------------------------------------------------------------------------"""

#Let's take one text message and get its bag-of-words counts as a vector:

bow_transform = CountVectorizer().fit(train_set['emails'])
print(len(bow_transform.vocabulary_))

#Can use .transform on the Bag-of-Words (bow) transformed object and transform the entire DataFrame of messages. 
message4=train_set['emails'][3]
print(message4)


#vector representation:

bow4=bow_transform.transform([message4])
print(bow4)
print(bow4.shape)

email_bow = bow_transform.transform(train_set['emails'])
print('Shape of Sparse Matrix: ',email_bow.shape) # 2272 emails now has 22324 Bag-of-Words
print('Amount of non-zero occurences:',email_bow.nnz) # non-zero occurences 152198

sparsity =(100.0 * email_bow.nnz/(email_bow.shape[0]*email_bow.shape[1]))
print('sparsity:{}'.format(round(sparsity))) # good sparsity is 0 


# Now lets use tfidfTransformer to weight the words
tfidf_transformer=TfidfTransformer().fit(email_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

email_tfidf=tfidf_transformer.transform(email_bow)
print(email_tfidf.shape)


"""----------------------------------------------------------------------------------------------------------------------------""" 
"""******************************************************* Model Building ************************************************""" 
"""-----------------------------------------------------------------------------------------------------------------------------"""
#Now that our words are weighted we can finally train our classifier.
# Instead of 'just' fitting the pipeline on the training
# data, do cross-validation too so that you know if it's
# overfitting.
# This returns an array of values, each having the score 
# for an individual run.

# Supervised classification    
n_folds = 5
def f1_cv(model):
    kf = KFold(n_folds, shuffle = True, random_state = 29).get_n_splits(email_tfidf)
    f1 = cross_val_score(model,email_tfidf,train_set['target'], scoring = 'f1', cv = kf )
    return (f1)

# models 
spam_model_NB = MultinomialNB(alpha = .2).fit(email_tfidf,train_set['target'])
spam_model_svc = SVC(kernel = 'sigmoid', gamma = 1.0).fit(email_tfidf,train_set['target'])
spam_model_rf = RandomForestClassifier(n_estimators = 31, random_state = 32).fit(email_tfidf,train_set['target'])
spam_model_GB = GradientBoostingClassifier( n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt',min_samples_leaf=15, min_samples_split=10, random_state =5).fit(email_tfidf,train_set['target'])
spam_model_dt = DecisionTreeClassifier().fit(email_tfidf,train_set['target'])


# lets test them
print('predicted:',spam_model_NB.predict(tfidf4)[0])
print('expected:',train_set.target[3])


print('predicted:',spam_model_svc.predict(tfidf4)[0])
print('expected:',train_set.target[3])


print('predicted:',spam_model_rf.predict(tfidf4)[0])
print('expected:',train_set.target[3])


print('predicted:',spam_model_GB.predict(tfidf4)[0])
print('expected:',train_set.target[3])



print('predicted:',spam_model_dt.predict(tfidf4)[0])
print('expected:',train_set.target[3])

### just to check that the models are all predicting correctly

"""----------------------------------------------------------------------------------------------------------------------------""" 
"""******************************************************* Model Testing ************************************************""" 
"""-----------------------------------------------------------------------------------------------------------------------------"""


result = f1_cv(spam_model_svc)
print ('\nSVC score: {:4f}({:4f})\n'.format(result.mean(), result.std()))


result = f1_cv(spam_model_rf)
print ('\nRandomForest score: {:4f}({:4f})\n'.format(result.mean(), result.std()))


result = f1_cv(spam_model_NB)
print ('\nMultinomial NB score: {:4f}({:4f})\n'.format(result.mean(), result.std())) 

result = f1_cv(spam_model_GB)
print ('\nGradient Boosting GB score: {:4f}({:4f})\n'.format(result.mean(), result.std())) 


result = f1_cv(spam_model_dt)
print ('\nDecision Tree DT score: {:4f}({:4f})\n'.format(result.mean(), result.std())) 



#Now we want to determine how well our model will do overall on the entire dataset. Let's begin by getting all the predictions:

all_predictions = spam_model_svc.predict(email_tfidf)
print(all_predictions)

#SciKit Learn's built-in classification report, which returns precision, recall, f1-score, and a column for support (meaning how many cases supported that classification)


print(classification_report(train_set['target'],all_predictions))
print(confusion_matrix(train_set['target'],all_predictions))

# as expected looks fairly accurate fitted on the train set 

"""----------------------------------------------------------------------------------------------------------------------------""" 
"""******************************************************* Pipeline, Model Testing and Validation ************************************************""" 
"""-----------------------------------------------------------------------------------------------------------------------------"""

pipeline = Pipeline([
   ( 'bow',CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier',SVC(kernel = 'sigmoid', gamma = 1.0)),
])
# for pipeline use best model 
pipeline.fit(X_train,y_train)
filename = 'finalized_model.sav'
pickle.dump(pipeline, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
# result of 0.9774 

predictions = pipeline.predict(X_test)

print(classification_report(predictions, y_test))
# get an accuracy of 98% on test set. very good.

# lets build a confusion matrix to double check
confusionMatrix = confusion_matrix(y_test,predictions)
pd.DataFrame(data = confusionMatrix, columns = ['Predicted 0', 'Predicted 1'],
            index = ['Actual 0', 'Actual 1'])

# 526+425/974 = 0.97638




 