1.6 million tweets.ipynb is the file were Bernoulli_NB, LogisticRegression, SVC and vectoriser models were trained. Trained on a 1.6 million tweets dataset. 
Sentiment140dataset.py is the script version of the notebook

TestNaiveBayes.py is used to test NLTK's Naive Bayes Model with custom dataset. It was trained on the documents in the reviews folder.

https://drive.google.com/drive/folders/1TuPbXVF41TeiAUlzw1GQHeQP9WvWeNpi?usp=sharing
In here you can find the notebook.db file, which is the 1.6 million tweets.ipynb kernel saved, so you don't have to run it all again.
The other file is the 1.6 million tweets dataset. Comes from: https://www.kaggle.com/kazanova/sentiment140/data









Combining Algorithms with NLTK to create our classifier
Combining classifier algorithms is is a common technique, done by creating a sort
 of voting system, where each algorithm gets one vote, and the classification that 
 has the most votes is the chosen one.

Text classification
A fairly popular text classification task is to identify a body of text as either
spam or not spam, for things like email filters. In our case, we're going to try 
to create a sentiment analysis algorithm.

Naive Bayes will take every word in every review to find the most popular words used. 
Then, out of those most popular words we'll see which one appeared on positive
or negative connotations. Finally, we'll search for those words for whichever has
more positive or negative and that's how will classify.

CUSTOM DATASET GIVES BETTER PREDICTIONS RATHER THAN USING THE MOVIEW REVIEWS