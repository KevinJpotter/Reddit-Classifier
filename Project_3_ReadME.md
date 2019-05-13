
# Project 3 ReadME:  Reddit Challenge

### Problem
Build a model that takes in information on a reddit post and predicts if the post belongs to a specific subreddit.



### Data Gathering
[code](api_scrape_code.ipynb)

The reddits I chose were conspiracy and history because I thought they would have an interesting relationship. I used the API offered by reddit with the goal of collecting 1000 posts for each subreddit. I created a loop that would send a request to the API every few seconds, each time grabbing 100 posts and storing the .json in a list.

After looking though the .json file I extracted the titles, category, and body (where available) for all of the posts and stored my findings in a dictionary, then to a csv to be used to explore.



## Natural Language Processing
[code](nlp_code.ipynb)

I loaded the .csv with all of my information into a data frame to have a look. About half of the posts I collected for conspiracy had a body available and all but 60 of the posts for history had a body. I decided to make 4 columns to the later test and prepare fairly would be the best route. 

I started with my first column as just the title of the post, I then made a columns which combined the title and body into a separate feature I would use for testing. I wanted to use the body of the post and this seemed like the best way to use the information that was there but, would not limit the scope of the model. 

With these two columns created two more condensed versions of these columns. I ran both through a porter stemmer, I also got rid of all punctuation and special characters in the process to help to limit the amount of features that would then be used.

I also did a small amount of modeling toward the end of the book to be able to get a sense for what would be worth while params to iterate over when it was time to model.

I exported a .csv file with the data I would be using for model and comparison.


|Feature|Data Type|Data Description|
|---|---|---|
|title|string|the title of every post|
|combine|string|the body combined with the title|
|stem_title|string|the title of the reddit stemmed and punctuation removed|
|stem_combined|string|the title and body of the reddit stemmed and punctuation removed|
|cat_nums|integer| 1 = history 0 = conspiracy|


### Modeling
[code](model_code.ipynb)

In order to be able to use multiple I created a function that took in the parameters X, y, a model, the vectorizers I would use, and the parameters of the model I would like to grid search over. The return would be a list of dictionaries that included a short summary of the model, the transformer used, the accuracy score for testing data, the accuracy score for testing data, and the best estimators from the model that could be used for further analysis.

The vectorizers I used for every model were count vectorizer and TFIDF vectorizer with removing the stop words. I over 10 different model on the title column then used the top three scoring models for further investigation. The top three performing models were multinomial naive bayes, k-nearest neighbors, and logistic regression with a bagging classifier.

I then ran each of these classifiers over on each of the X features described in the dictionary above further optimizing performance with tuning of parameters.


### Conclusions / Analysis

In conclusion the model that performed the best was multinomial naive bayes with and accuracy score of 92% on unseen data. This model used the stemmed combined feature, had good amount of variance to the training data, but ultimately got the most predictions correct. When using the stemmed columns I notice the recall going up substantially compared to the title column models alone. I believe this to be because so many more of the history columns had bodies in turn the model would predict more posts to be history.

Before use of this model I would like to understand more about is specific purpose and look into further of eliminating certain key words that may make the model less accurate in a short period of time. Some of the words that had high coeficients on the logistic regression model were from trending stories which soon would make the model predict the wrong thing or make posts harder to identify. It would also help to limit the computing power necessary to complete the task if this is being done on a large scale.
