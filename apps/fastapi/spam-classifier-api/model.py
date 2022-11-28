# import dependencies
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# load spam / ham dataset
def load_data():

    # transform dataset to dataframe
    PATH = 'SMSSpamCollection'
    df = pd.read_csv(PATH, delimiter = "\t", names=["classe", "message"])

    # define the X and the y columns
    X = df['message']
    y = df['classe']

    return X, y

# split data in training and test sets
def split_data(X, y):

    # test size must be between 0 and 1
    ntest = 2000/(3572+2000)

    # split data in training and test set
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=ntest, random_state=0)

    return X_train, y_train

# define spam classifier model
def spam_classifier_model(Xtrain, ytrain):

    # logistic regression
    model_logistic_regression = LogisticRegression()
    model_logistic_regression = model_logistic_regression.fit(Xtrain, ytrain)

    # absolute value of the coefficients
    coeff = model_logistic_regression.coef_
    coef_abs = np.abs(coeff)

    # quantiles of the coefficients (absolute value)
    quantiles = np.quantile(coef_abs,[0, 0.25, 0.5, 0.75, 0.9, 1])

    # choose the first quartile (25%)
    index = np.where(coeff[0] > quantiles[1])
    newXtrain = Xtrain[:, index[0]]

    # create model
    model_logistic_regression = LogisticRegression()

    # model fit
    model_logistic_regression.fit(newXtrain, ytrain)

    return model_logistic_regression, index

# extract input and output data
data_input, data_output = load_data()

# split data
X_train, ytrain = split_data(data_input, data_output)

# transform and fit training set
vectorizer = CountVectorizer(stop_words='english', binary=True, min_df=10)
Xtrain = vectorizer.fit_transform(X_train.tolist())
Xtrain = Xtrain.toarray()

# use the model and index for prediction
model_logistic_regression, index = spam_classifier_model(Xtrain, ytrain)

