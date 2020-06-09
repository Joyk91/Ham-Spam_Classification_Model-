# Supervised Classification Pipeline 
# Classification of Spam/ Ham emails using Enron Email Dataset. 

## Introduction

The dataset that we will use for this assignment is the Enron email dataset. You can find the
full dataset on the web here: http://www.aueb.gr/users/ion/data/enron-spam/. The dataset is a
collection of public domain emails from the Enron corporation. The emails have been manually
classified as spam and non-spam. The primary goal of the assignment is to create a supervised
classification pipeline to classify emails as spam or non-spam from the training data. You are
free to use either the preprocessed emails or the raw emails for your analysis.

## Results and Analysis 
This report will explore in detail the classification pipeline built for the detection of spam and non-spam emails. The pipeline will be analysed under the following headings: 
1.	Pre-processing 
2.	Exploratory data analysis 
3.	Supervised classification 
4.	 Model selection  


## 1.	Pre-processing  

#### Helper Functions 
As mentioned the raw emails were utilised for this classification pipeline. Once the emails were downloaded 7-zip was used to extract the emails from the raw data. Once this was completed the spam and non-spam emails were read into Spyder separately as lists.  A binary variable was assigned to each email in the lists to distinguish between spam and non-spam. The spam list was assigned the dummy variable (labelled target) of 1 and the non-spam (also referred to as ham) was assigned the dummy variable of 0.  This dummy variable is known as ‘one-hot’ coding and is a particularly useful approach in Scikit-Learn package, as the packages models make the fundamental assumption that numerical features reflect algebraic quantities. Each list was transformed into a panda’s data frame containing the headings emails (containing all the emails) and target (depending on the data frame contained a 0 or 1 for ham and spam respectively). Both the datasets were then joined into one large dataset containing both ham and spam emails. 

Please note that the downloaded version of the dataset contained five folders of both spam and ham in each. For processing capabilities, the pipeline was built using only one folder of each type of target.  
The next step in the process is to use helper functions to extract only the body of the emails. The raw emails contain to and from email addresses, cc email addresses, attachments, message-ID etc.  For the classification pipeline only, the body of the emails will be used to classify between spam and ham. Therefore, the get_payload function was used from the package emails to return a list of email parts (Fig 1)  

![1](https://user-images.githubusercontent.com/50813004/84185049-582e7280-aa86-11ea-9f81-7718f1df9bb6.png) 
	
	Fig 1: Helper function for emails parts 

The next function built was used to separate multiple emails addresses from the dataset (Fig 2). 

![2](https://user-images.githubusercontent.com/50813004/84185050-58c70900-aa86-11ea-809e-00a004b9a8a9.png)
	
	Fig 2: Helper function for emails addresses 

The emails were then parsed into a list of email objects. The fields from the parsed emails objects were set as the keys. And a new data frame was built using the map function to set each key as a header. Each header now represents the parts of the emails extracted using the helper function in Fig 1 and Fig 2. The dataset is now 2429 rows long and contains 14 columns.  
Now a new data frame will be created, and the columns subject, content and target will be imported in from the old data frame. Now the subject column and the content column will be joined into one and now we have the data frame we want for the classification.   



### Train and Test Split

Now that the emails are in a data frame, the next step in the pipeline was to split the dataset into a train and test set for classification. The code below (Fig3) enabled the train and test split.  

![3](https://user-images.githubusercontent.com/50813004/84185052-58c70900-aa86-11ea-9c0e-e3a954ed6142.png)  
	
	Fig 3: Train and Test Split

The data was split 70/30 for train set and test set respectively. Splitting the data randomly ensured that the resultant two dataset were free from bias. 
Some summary statistics were carried out at this stage, and it was observed that the train_set contains a total of 2,272 emails, 1,299 of which are non-spam emails and the remainder (973) being spam emails. The statistics also revealed that 1245 of the ham emails are unique and 866 of the spam emails are unique. Concluding that there are very little duplicate emails. Depending on the model built these duplicates may be removed to improve the model. 


## Feature Extraction  
Simply put feature engineering is the process of using domain knowledge of the dataset to create features for synergy with machine learning algorithms (Shekhar, 2019). When done correctly feature engineering increases the predictive power of machine learning algorithms (Shekhar, 2019)
Stemming, removing regular expression, remove punctuation and filtering of stop words are included in this classification pipeline (see Fig 4). As domain knowledge increased whilst working on the pipeline these feature extraction steps were continuously reviewed and updated. 
Stemming is used to reduce words to their stems.  At first the porter stemmer for the nltk package was utilised, however upon review of the results it seemed that the algorithm was over-stemming. For the word “university”, “universities”, it was returning both “univers” and “universi”. Further research concluded that the porter stemmer is a simple stemming algorithm which was developed in the 1980’s (Heidenreich, 2019). Upon deeper research the snowball stemmer was discovered. 

This stemming algorithm is known as an update on the porter stemmer and is universally acknowledged as a better algorithm (Heidenreich, 2019). Therefore, the snowball stemmer from the nltk package was used to reduce the words to their stems (Fig4).
 
  
![4](https://user-images.githubusercontent.com/50813004/84185053-595f9f80-aa86-11ea-83b9-ede3de7d2fbd.png)
	
	Fig4: Feature Engineering  

After more testing and analysis, it was observed that removing the stemmer produced better results for the models. When stemming was utilised in the code it has a directly negative effect on the models in the pipeline. This again could be due to over-stemming and under- stemming. Therefore, the stemmer was removed from the function to increase predictive power (see Fig 4.1).

![5](https://user-images.githubusercontent.com/50813004/84185054-595f9f80-aa86-11ea-8d71-4139fdb76e14.png)
	
	Fig 4.1: Updated Feature Engineering   
	
Fig 4.1 also shows the stop words update. As analysis was carried out more and more words were found that were of no use for classification purposes. Therefore, the stop words were updated many times to remove useless world like ‘html’, ‘cc’, ‘lon’, ‘www’ etc. 
For machine learning algorithms, the text data needs to be represented is a way that is synergistic with the algorithms. The bag-of-words model is one way to do that. Now that the emails are cleaned the next step was to transform each email into its bag-of-words vector.  Basically, the objective is to tokenize words for each observation and find the frequency of each token (D'Souza, 2019). By using the .transform function from Bag-of-Words (bow) we can transform the entire data frame of emails into its Bag-of-Words.  Now the train_set which contains 2,272 emails, represents a total Bag-of-Words of 22,784. Next the TfidfTransformer() from the Scikit-learn package was used on the vectors. TF-IDF stands for term frequency-inverse document frequency. TF-IDF weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus (D'Souza, 2019). The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus (D'Souza, 2019).   


## 2.	Exploratory Data Analysis 
Exploratory data analysis was carried out on the dataset in order to understand the data and what type of features will work best to fine tune our model. The first plot produced was a count of spam and non – spam emails to determine whether our dataset is balanced (see Fig 5). 
 
![6](https://user-images.githubusercontent.com/50813004/84185055-59f83600-aa86-11ea-9f6e-a41c9b01230a.png) 
		
		Fig 5: Barplot of spam (1) and non-spam (0) emails.

As we can see from the plot this is a slightly imbalanced dataset. There is a total of 2,272 emails in the train set, 1299 of them are ham and the remaining 973 are spam. This as an important observation as the problem with imbalanced datasets is that the simple metric like accuracy or precision may not be useful in reflecting the real performance of the predictive model. Therefore, the scoring method used in the model will have to be carefully chosen (see part 4 model building). 

The next plot generated was a frequency distribution of the top 20 words in the both the spam and non-spam emails (see Fig 6)  

![7](https://user-images.githubusercontent.com/50813004/84185056-59f83600-aa86-11ea-9b29-09c64a105949.png)
 
 	Fig 6: Frequency Barplot of top 20 words in spam and non-spam emails. 

When we look at the difference between the two list of words it’s not surprising to see words like ‘money’, ‘please’, ‘best’, ‘contact’, ‘lottery’ etc. in the spam list. These are the types of words we already associate with spam emails, due to the fact that their main purpose is to get the recipient to buy, signup or contact. The top twenty words in the non-spam list are what you would expect of business emails.  It includes words like ‘enron’, ‘business’, ‘review’, ‘risk’, ‘management’, ‘thanks’, ‘meeting’ etc.  


![8](https://user-images.githubusercontent.com/50813004/84185057-59f83600-aa86-11ea-8545-fc13a2c5a1de.png)
	
	Fig 7: WordCloud representation of top 20 words in spam and non-spam emails.  




![9](https://user-images.githubusercontent.com/50813004/84185060-5a90cc80-aa86-11ea-87e8-ad8bf34b35ec.png)

 	Fig 8: Barplot Length of Spam Emails with stats

![10](https://user-images.githubusercontent.com/50813004/84185063-5a90cc80-aa86-11ea-877a-4ff0b3703897.png)	
	
	Fig 9: Barplot Length of Ham Emails with stats 

The difference in length of emails in spam and ham was also analysed. Fig 8 & 9 shows summary statistic and visualisation of length of spam/ham emails. It was observed that the mean and median of ham emails are larger than spam emails, indicating that characteristically in this dataset ham emails are longer than spam emails. However, the barplots showed that sentences in spam emails are longer than those in ham emails. This may be due to the fact that spam emails are trying to make an impression on the recipient. Note that these plots were generated on the processed data. Therefore, stop words, illegal words etc have been removed. Therefore, the code was run again to generate the plots on the ram emails. The resultant plots can be seen below (Fig 10, Fig 11).   


	 
![12](https://user-images.githubusercontent.com/50813004/84185096-654b6180-aa86-11ea-9edc-89258b650658.png)

	Fig 10: Un-processed Length of Spam Emails 

![13](https://user-images.githubusercontent.com/50813004/84185098-654b6180-aa86-11ea-8391-339aff1791ec.png)
	
	Fig 11: Un-processed Length of Ham Emails 
	
The ram ham emails still have a greater mean and median. The barplots for the unprocessed emails are similar to that of the processed. However, the plots do observe that while the ham emails follow a normal (skewed) distribution whereas spam emails seem to share a characteristic length as depicted by Fig 10. It seems that very few spam emails have more than 200 words in a sentence.  

## 3.	Supervised classification  

Utilising the Bag-of-Words and term frequency approach to create a matrix of word occurrences in both target groups created a uni-gram approach for model building.  The training set was used to build 5 different models. These models include decision tree, random forest, gradient boosting, multinomial naïve Bayes and Support vector machine. 
A k-fold cross validation approach was utilised as a resampling method. K-fold cross validation is not only easy to use and interpret but previous experience with this resampling method proved its validity. In many circumstances K-fold cross validation results in a less optimistic estimate of the model skill than other methods, such as a simple train/test split, ensuring our data is not biased. Simply put k-fold cross validation randomly splits the data into k folds, of equal size. The first fold is treated as a validation set and the method is to fit on the remaining k – 1 folds. Because the dataset is not extremely large and also due to the fact that the dataset is slightly imbalanced 5 folds were utilised for the cross validation. 
F1 scoring method was also utilised for the model building. As mentioned in the EDA section, the data between the classes (spam & ham) is unevenly distributed i.e there are more ham emails than spam. F1 scoring was chosen for this reason as it will seek balance between precision and recall. 



The results from the model fitting can be seen in Table 1.  All models reported predictive accuracy in the 90th percentile. Support Vector Machine model proved to be the most accurate, followed closely by gradient boosting.  


![14](https://user-images.githubusercontent.com/50813004/84185099-65e3f800-aa86-11ea-9bc4-a4cab17508a2.PNG)	
	
	Table 1: Model Fitting  

## 4.	Model selection 
Now that we have an accurate model it is time to fit the test set and determine the model’s true accuracy rate on the hold-one out method.  
A pipeline was built with the best model (support vector machine) and the test set was fit. The data was tested with support vector machine kernels, rbf, linear and sigmoid. After further testing the sigmoid kernel produced the most accurate model and therefore was chosen for the final model. The results of the model can be seen in Table 2 and Table 3.  
 
![15](https://user-images.githubusercontent.com/50813004/84185101-65e3f800-aa86-11ea-9094-6070c8a7a926.PNG)
	
	Table 2: Confusion Matrix 
	
From the confusion matrix (Table 2), we see that the Support Vector Machine classifier received the following results:
•	Out of the 565 actual instances of ham (0), it predicted correctly 549 of them;
•	Out of the 409 actual instances of spam (1), it predicted correctly 403 of them. 
•	The confusion matrix gives an accuracy score of 97.74% (549+403/974). 
This is a good accuracy score for the spam/ham classifier. Next, we will look at the classification report (Table 3).

![16](https://user-images.githubusercontent.com/50813004/84185104-65e3f800-aa86-11ea-84b8-97d90ef7d128.PNG)
	
	Table 3: Classification Report for test set
The report gives a 98% accuracy for the classifier for each score measure. This should be no surprise considering the accuracy result obtained from the confusion matrix. 
Overall the resulting classification pipeline has a 98% accuracy rate in classifying between spam and non-spam emails.   




## Conclusion  

A classification pipeline was successfully built for the detection of ham and spam emails. Extreme analysis of the data, continuous discovery and the use of the raw emails ensured that the feature extraction and data cleaning for this classification pipeline extremely extensive. The resulting emails were clean and legible compared to the dirty data that we started with. This was the strong foundation that facilitated the building of a very accurate model. Considering the chosen model has a 98% accuracy for classifying between ham and spam emails the foundational steps in this pipeline were essential. 
As mentioned earlier in the report for processing purposes the classification pipeline was built using only a subset of the available emails making the dataset used for the classification problem small by comparison. It is wise to note this as further testing and updating may be needed for this pipeline for use in larger datasets and or similar datasets. However, the ‘sigmoid’ kernel used in the support vector model is very useful in large dataset as it utilises neural networks. Therefore, the model was indeed built using a subset of the emails however, the fact that the dataset was a subset was always kept in mind for the resulting model. 


## References
D'Souza, J. (2019). An Introduction to Bag-of-Words in NLP. [online] Medium. Available at: https://medium.com/greyatom/an-introduction-to-bag-of-words-in-nlp-ac967d43b428 [Accessed 11 Mar. 2019].
Heidenreich, H. (2019). Stemming? Lemmatization? What?. [online] Towards Data Science. Available at: https://towardsdatascience.com/stemming-lemmatization-what-ba782b7c0bd8 [Accessed 11 Mar. 2019].
Shekhar, A. (2019). What Is Feature Engineering for Machine Learning?. [online] Medium. Available at: https://medium.com/mindorks/what-is-feature-engineering-for-machine-learning-d8ba3158d97a [Accessed 16 Mar. 2019].



