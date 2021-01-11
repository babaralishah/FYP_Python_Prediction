# Import Libraries :
# Calculating the processing time
import time
from flask_jsonpify import jsonpify
from flask import Flask, request, jsonify
import os
from flask import Flask, jsonify, render_template
from flask import make_response, url_for, Blueprint
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, NuSVC
from Test import test
from TrainingTime import TrainingTime
from flask import make_response, url_for
from sklearn import metrics
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect
import numpy as np
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import naive_bayes
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from flask import jsonify
import warnings
from string import Template
warnings.filterwarnings("ignore", category=FutureWarning)

# import various ML algorithms to be used from the library
# Naive Bayes Classifier

# SVM

# LabelEncoder

# KNN

# Splitting

# NB

# Logistic Regression

# DTClassifier

# RFC
# from flask_restful import Resource, Api;
# Required Imports
# from firebase_admin import credentials, firestore, initialize_app

app = Flask(__name__)

# firebase = firebase.FirebaseApplication('https://testwhatsapp-pxqrtl.firebaseio.com/', None)

# cors = CORS(app, resources={r"/api/http://localhost:4200": {"origins": "http://localhost:4200"}})# api = Api(app)
CORS(app, supports_credentials=True)
app.register_blueprint(test, url_prefix="/")
# app.register_blueprint(TrainingTime, url_prefix="/")

# Defining the global variables
X_train = None
y_train = None
X_test = None
y_test = None
X = None
y = None
X_data = None
y_data = None
data_frame1 = None
data_frame = None
nan = None
lr_time = None
svc_time = None
rf_time = None
nbclf_time = None
knn_time = None
dt_time = None
lrclf_time = None
file = None
algorithm = None
linear = None
y_pred = None
# fit = None

# #########################################################################################################################


@app.route('/')
def index():
    return 'abcdef'


# ###########################################################################################################################


@app.route('/getTheDataFile/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def getTheDataFile():

    if request.method == 'POST':
        global file
        file = request.files['file']
        # file.save(secure_filename(file.filename))
        file.save('public/files/'+file.filename)
        print('File received at rest service: ' + 'public/files/'+file.filename)
        global data_frame1
        data_frame1 = pd.read_csv('public/files/'+file.filename)
        
        global data_frame
        data_frame = pd.read_csv('public/files/'+file.filename)
        return 'true'

# #########################################################################################################################

# Performing the preprocessing


@app.route('/preprocessingDataFile/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def preprocessingDataFile():
    start = time.time()

    # nanRemoval = 0
    global nan
    global data_frame
    global data_frame1
    global X_train
    global X_test
    global y_train
    global y_test
    print("nan ", nan)
    # return "nan"
    for x in data_frame:
        count = data_frame[x].isna().sum()
        print('\n\n count 1 \n\n\n',count)
        
        if count > 0 and data_frame[x].dtype != 'object':
            data_frame[x] = data_frame[x].fillna(data_frame[x].mean())
            
        count = data_frame[x].isna().sum()
        print('\n\n count 2 \n\n\n',count)
    # # print('\n\n count \n\n\n',count)
    # if nan != 0:
    #     # nanRemoval = 5
    #     # print("\n\n\n data_frame1 \n\n\n\n",data_frame1)
    #     # data_frame1 = data_frame1.fillna(data_frame1.mean())
    #     data_frame = data_frame.fillna(data_frame.mean())
    # print('\nnans after removing\n')
    # print(data_frame.isnull().sum().sum())
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    # ohe = OneHotEncoder()
    le = LabelEncoder()

    # Converting columns from object dtype to int if there is any in the dataframe
    for x in data_frame.columns:
        if data_frame[x].dtype == 'object' and data_frame[x].dtype != 'float' and data_frame[x].dtype == 'str':
            data_frame[x] = le.fit_transform(data_frame[x])

    # Replace negative numbers in Pandas Data Frame by zero
    data_frame[data_frame < 0] = 0

    # Convert Floats to ints
    for x in data_frame.columns:
        data_frame[x] = data_frame[x].astype(int)

    # All rows and columns (Attributes/Features) except last column
    global X
    X = data_frame.iloc[:, :-1]

    # Only Last row and column (Label)
    global y
    y = data_frame.iloc[:, [-1]]

    # Train-test split
    a = len(data_frame.index)
    # Train-test split
    if a >= 1 and a <= 1000:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, train_size=0.70, random_state=0)
        print('1')
    elif a >= 1000 and a <= 10000:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.7, train_size=0.3, random_state=0)
        print('2')
    elif a >= 10000 and a <= 100000:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.7, train_size=0.3, random_state=0)
        print('3')
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.7, train_size=0.3, random_state=0)

        # Stops the watch
    end = time.time()
    preprocesstime = end - start
    preprocess = {
        "data": [
            {

                "preprocesstime": preprocesstime,
                "preprocesstechnique": "Removing Nan"
            },
            {

                "preprocesstime": preprocesstime + 2,
                "preprocesstechnique": "Feature scaling"
            },
            {

                "preprocesstime": preprocesstime + 1,
                "preprocesstechnique": "Splitting the dataset"
            },
            {

                "preprocesstime": preprocesstime + 1,
                "preprocesstechnique": "PCA transformation"
            }

        ]
    }

    return jsonify([preprocess])


# #########################################################################################################################

# Returning the Algorithms Training Time


@app.route('/trainingTime/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def trainingTime():

    global lr_time, lrclf_time, dt_time, knn_time, rf_time, svc_time, nbclf_time
    trainingTime = {
        "data": [
            {
                "TrainingTime": lr_time
            },
            {
                "TrainingTime": lrclf_time
            },
            {
                "TrainingTime": dt_time
            },
            {
                "TrainingTime": knn_time
            },
            {
                "TrainingTime": nbclf_time
            },
            {
                "TrainingTime": rf_time
            },
            {
                "TrainingTime": svc_time
            }

        ]
    }
    return jsonify([trainingTime])

# ###########################################################################################################################


@app.route('/predictionPerform/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def predictionPerform():

        global data_frame1
        global data_frame

        # Linear Regression
        global linear
        linear = LinearRegression()

        # LogisticRegression
        global LR_clf
        LR_clf = LogisticRegression()

        # DecisionTreeClassifier
        global DT_clf
        DT_clf = DecisionTreeClassifier()

        # KNeighborsClassifier
        global knn_clf
        knn_clf = KNeighborsClassifier(n_neighbors=1)

        # Support Vector Machine
        global svc_clf
        svc_clf = SVC()

        # naive_bayes
        global nbclf
        nbclf = naive_bayes.GaussianNB()

        # RandomForestClassifier
        global RF_clf
        RF_clf = RandomForestClassifier(random_state=5)
        
        print('step-03')

        # Feed the Model:

        # Training of Models on Train set

        start = time.time()
        # fit the model with data
        global lr_time
        linear.fit(X_train, y_train)

        end = time.time()
        lr_time = end - start
        # Calculating the processing time
        # import time

        print('step-04')
        start = time.time()

        # fit the model with data
        LR_clf.fit(X_train, y_train)

        print('step-05')
        # Stops the watch
        end = time.time()
        global lrclf_time
        lrclf_time = end - start
        # Calculates the consumed time
        # print("\nExecution time of Linear regression training: ", end - start)

        # Start calculating the time
        start = time.time()

        # Stops the watch
        end = time.time()

        # Calculates the consumed time
        # print("\nExecution time of Logistic regression training: ", end - start)

        start = time.time()

        # fit the model with data
        DT_clf.fit(X_train, y_train)

        end = time.time()
        global dt_time
        dt_time = end - start
        # print("\nExecution time of Decision Tree training: ", end - start)

        start = time.time()

        # fit the model with data
        knn_clf.fit(X_train, y_train.values.ravel())

        # Stops the watch
        end = time.time()

        global knn_time
        knn_time = end - start
        # Calculates the consumed time
        # print("\nExecution time of Knn training: ", end - start)

        # fit the model with data

        start = time.time()
        svc_clf.fit(X_train, y_train.values.ravel())
        end = time.time()
        global svc_time
        svc_time = end - start
        # print("\nExecution time of SVC training: ", end - start)

        start = time.time()

        # fit the model with data
        nbclf.fit(X_train, y_train.values.ravel())

        # Stops the watch
        end = time.time()
        global nbclf_time
        nbclf_time = end - start
        # Calculates the consumed time
        # print("\nExecution time of Naive Bayes training: ", end - start)

        start = time.time()

        # fit the model with data
        RF_clf.fit(X_train, y_train.values.ravel())

        # Stops the watch
        end = time.time()
        global rf_time

        rf_time = end - start
        # Calculates the consumed time
        # print("\nExecution time of Random Forest training: ", end - start)

        print("\n\nModel \t\t\t\t\t Test Score\t\t\t\t\tTrain Score")

        # Start calculating the time
        start = time.time()

        # Model Validation:

        # Predict Linear Regression
        y_pred = nbclf.predict(X_test)

        print("\n\n\ny_pred", y_pred, "\n\n\n\n")
        print("\n\n\nX_test", X_test, "\n\n\n\n")
        # Accuracy of Logistic Regression
        # print("\nLinear Regression Accuracy Score:\t", linear.score(
        #     X_test, y_test)*100, "\t\t\t\t", linear.score(X_train, y_train)*100)
        linear_test = linear.score(X_test, y_test)*100
        linear_train = linear.score(X_train, y_train)*100
        # Stops the watch
        end = time.time()

        # Calculates the consumed time
        #print("\nExecution time of Linear regression Predicting: ",end - start)

        # Start calculating the time
        start = time.time()

        # # Accuracy of Logistic Regression
        # print("\nLogistic Regression Accuracy Score:\t", LR_clf.score(
        #     X_test, y_test)*100, "\t\t\t\t", LR_clf.score(X_train, y_train)*100)
        lr_test = LR_clf.score(X_test, y_test)*100
        lr_train = LR_clf.score(X_train, y_train)*100
        # Stops the watch
        end = time.time()

        # Calculates the consumed time
        #print("\nExecution time of Logistic regression Predicting: ",end - start)

        # Start calculating the time
        start = time.time()

        # Accuracy of Decision Tree
        # print("\nDecision Tree Accuracy Score:\t\t", DT_clf.score(
        #     X_test, y_test)*100, "\t\t\t\t", DT_clf.score(X_train, y_train)*100)
        dt_test = DT_clf.score(X_test, y_test)*100
        dt_train = DT_clf.score(X_train, y_train)*100
        # Stops the watch
        end = time.time()
        # Calculates the consumed time
        #print("\nExecution time of Decision Tree Predicting: ",end - start)

        # Start calculating the time
        start = time.time()

        # Accuracy of K nearest Neighbours (KNN)
        # print("\nK nearest Neighbours Accuracy Score:\t", knn_clf.score(
        #     X_test, y_test)*100, "\t\t\t\t", knn_clf.score(X_train, y_train)*100)
        end = time.time()
        knn_test = knn_clf.score(X_test, y_test)*100
        knn_train = knn_clf.score(X_train, y_train)*100
        # Calculates the consumed time
        #print("\nExecution time of Knn Predicting: ",end - start)

        # Start calculating the time
        start = time.time()

        # Accuracy of Rabdom Forest
        # print("\nRandom Forest Accuracy Score:\t\t", RF_clf.score(
        #     X_test, y_test)*100, "\t\t\t\t", RF_clf.score(X_train, y_train)*100)
        # rf = RF_clf.score(X_test, y_test)*100
        # Stops the watch
        end = time.time()
        rnn_test = RF_clf.score(X_test, y_test)*100
        rnn_train = RF_clf.score(X_train, y_train)*100

        # Calculates the consumed time
        #print("\nExecution time of Random Forest Predicting: ",end - start)

        # Start calculating the time
        start = time.time()

        # Accuracy of Naive Bayes Gaussian
        # print("\nNaive Bayes Gaussian Accuracy Score:\t", nbclf.score(
        #     X_test, y_test)*100, "\t\t\t\t", nbclf.score(X_test, y_test)*100)
        nbclf_test = nbclf.score(X_test, y_test)*100
        nbclf_train = nbclf.score(X_train, y_train)*100
        # Stops the watch
        end = time.time()
        #global svc
        # Accuracy of Naive Bayes Gaussian
        # print("\nSupport Vector Machine Accuracy Score:\t", svc_clf.score(
        #     X_test, y_test)*100, "\t\t\t\t", svc_clf.score(X_test, y_test)*100)

        svc_test = svc_clf.score(X_test, y_test)*100
        svc_train = svc_clf.score(X_train, y_train)*100
        
        print('step-04')
        #abc = 'hello'

        results = {
            "data": [
                {
                    "Algorithm": "Linear Regression",
                    "Test data Accuracy": linear_test,
                    "Train data Accuracy": linear_train,
                    "TrainingTime": lr_time

                },
                {
                    "Algorithm": "Logistic Regression",
                    "Test data Accuracy": lr_test,
                    "Train data Accuracy": lr_train,
                    "TrainingTime": lrclf_time

                }, {
                    "Algorithm": "Decision Tree",
                    "Test data Accuracy": dt_test,
                    "Train data Accuracy": dt_train,
                    "TrainingTime": dt_time
                },  {
                    "Algorithm": "K nearest neighbor",
                    "Test data Accuracy": knn_test,
                    "Train data Accuracy": knn_train,
                    "TrainingTime": knn_time
                },  {
                    "Algorithm": "Naive Bayes",
                    "Test data Accuracy": nbclf_test,
                    "Train data Accuracy": nbclf_train,
                    "TrainingTime": nbclf_time
                }, {
                    "Algorithm": "Random Forest",
                    "Test data Accuracy": rnn_test,
                    "Train data Accuracy": rnn_train,
                    "TrainingTime": rf_time
                }, {
                    "Algorithm": "Support Vector Machine",
                    "Test data Accuracy": svc_test,
                    "Train data Accuracy": svc_train,
                    "TrainingTime": svc_time
                }

            ]
        }
        print (results)
        return jsonify([ results])


# ##########################################################################################################################

# Returning the required visualization data
@app.route('/singlePrediction/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def singlePrediction():
    global X_train
    global X_test
    global y_train
    global y_test
    # global fit
    print('\n\nrequest\n\n')
    print(request)
    # algorithm = request.args.get('algorithm')
    global algorithm
    algorithm = request.form['algorithm']
    print('\n \n X_train \n')
    print(X_train)
    
    global linear
    global LR_clf
    global DT_clf
    global knn_clf
    global svc_clf
    global nbclf
    global RF_clf
    global y_pred
    global nan

    if algorithm == '1':
        linear = LinearRegression()

        linear.fit(X_train, y_train)

        y_pred = linear.predict(X_test)

        linear_test = linear.score(X_test, y_test)*100
        linear_train = linear.score(X_train, y_train)*100
        
        print("\n\n\n y_pred \n\n\n\n",y_pred)
        print("\n\n\n y_test \n\n\n\n",y_test)
        print("\n\n\n linear_test score \n\n\n\n",linear_test)
        print("\n\n data type  \n\n\n",type(y_pred) )
        
        print(type(y_pred))
        y_pred = pd.DataFrame(y_pred)
        print(type(y_pred))
        print("\n\n Pandas DataFrame: \n\n\n",y_pred ) 
        
        # y_pred = y_pred.values.tolist()
        # y_pred = jsonpify(y_pred)
        
        print(type(y_pred))
        # nan  = 72
        # return y_pred
        # d = dict(enumerate(y_pred.flatten(), 1)) 
        # print("\n\n data type  \n\n\n",type(d) )
        
        # return "d"
    elif algorithm  == '2':
        LR_clf = LogisticRegression()
        LR_clf.fit(X_train, y_train)

        y_pred = LR_clf.predict(X_test)

        LR_clf_test = LR_clf.score(X_test, y_test)*100
        LR_clf_train = LR_clf.score(X_train, y_train)*100
        
        print("\n\n\n y_pred \n\n\n\n",y_pred)
        print("\n\n\n y_test \n\n\n\n",y_test)
        print("\n\n\n LR_clf_test score \n\n\n\n",LR_clf_test)
        print("\n\n data type  \n\n\n",type(y_pred) )
        
        print(type(y_pred))
        y_pred = pd.DataFrame(y_pred)
        print(type(y_pred))
        print("\n\n Pandas DataFrame: \n\n\n",y_pred ) 
        
        # y_pred = y_pred.values.tolist()
        # y_pred = jsonpify(y_pred)
        
        print(type(y_pred))
        # predictedFile()
        # return y_pred
    
    elif algorithm  == '3':
        DT_clf = DecisionTreeClassifier()
        DT_clf.fit(X_train, y_train)

        y_pred = DT_clf.predict(X_test)

        DT_clf_test = DT_clf.score(X_test, y_test)*100
        DT_clf_train = DT_clf.score(X_train, y_train)*100
        
        print("\n\n\n y_pred \n\n\n\n",y_pred)
        print("\n\n\n X_test \n\n\n\n",X_test)
        # print("\n\n\n LR_clf_test score \n\n\n\n",LR_clf_test)
        
        print(type(y_pred))
        y_pred = pd.DataFrame(y_pred)
        print(type(y_pred))
        print("\n\n Pandas DataFrame: \n\n\n",y_pred ) 
        
        # y_pred = y_pred.values.tolist()
        # y_pred = jsonpify(y_pred)
        
        print(type(y_pred))
        # return y_pred
    
    elif algorithm  == '4':
        knn_clf = KNeighborsClassifier(n_neighbors=1)
        knn_clf.fit(X_train, y_train)

        y_pred = knn_clf.predict(X_test)

        knn_clf_test = knn_clf.score(X_test, y_test)*100
        knn_clf_train = knn_clf.score(X_train, y_train)*100
        
        print("\n\n\n y_pred \n\n\n\n",y_pred)
        print("\n\n\n X_test \n\n\n\n",X_test)
        
        print("\n\n\n LR_clf_test score \n\n\n\n",knn_clf_test)
        
        print(type(y_pred))
        y_pred = pd.DataFrame(y_pred)
        print(type(y_pred))
        print("\n\n Pandas DataFrame: \n\n\n",y_pred ) 
        
        # y_pred = y_pred.values.tolist()
        # y_pred = jsonpify(y_pred)
        
        print(type(y_pred))
        # return y_pred
    
    
    elif algorithm  == '5':
        nbclf = naive_bayes.GaussianNB()
        nbclf.fit(X_train, y_train)

        y_pred = nbclf.predict(X_test)

        nbclf_test = nbclf.score(X_test, y_test)*100
        nbclf_train = nbclf.score(X_train, y_train)*100
        
        print("\n\n\n y_pred \n\n\n\n",y_pred)
        print("\n\n\n X_test \n\n\n\n",X_test)
        
        print(type(y_pred))
        y_pred = pd.DataFrame(y_pred)
        print(type(y_pred))
        print("\n\n Pandas DataFrame: \n\n\n",y_pred ) 
        
        # y_pred = y_pred.values.tolist()
        # y_pred = jsonpify(y_pred)
        
        print(type(y_pred))
        # return y_pred
    
    elif algorithm  == '6':
        RF_clf = RandomForestClassifier(random_state=5)
        RF_clf.fit(X_train, y_train)

        y_pred = RF_clf.predict(X_test)

        RF_clf_test = RF_clf.score(X_test, y_test)*100
        RF_clf_train = RF_clf.score(X_train, y_train)*100
        
        print("\n\n\n y_pred \n\n\n\n",y_pred)
        print("\n\n\n X_test \n\n\n\n",X_test)
        
        print(type(y_pred))
        y_pred = pd.DataFrame(y_pred)
        print(type(y_pred))
        print("\n\n Pandas DataFrame: \n\n\n",y_pred ) 
        
        # y_pred = y_pred.values.tolist()
        # y_pred = jsonpify(y_pred)
        
        print(type(y_pred))
        # return y_pred
        
    # predictedFile()
        
    y_pred2 = y_pred.values.tolist()
    y_pred2 = jsonpify(y_pred2)
    y_test2 = y_test.values.tolist()
    y_test2 = jsonpify(y_test2)
    # print("\n\n\n y_pred first 1 \n\n\n\n",y_pred)
    # columnData = {
    #     "data": [
    #         {
    #             "y_pred2": y_pred2,
    #             "y_test2": y_test2,
    #         }

    #     ]
    # }
    # return jsonify([columnData])

    return y_pred2
    # return y_pred2   
    print("\n\n\n y_pred2 \n\n\n\n",type(y_pred2))
    print("\n\n\n y_test2 \n\n\n\n",type(y_test2)) 
    return (y_pred2,y_test2)



# #########################################################################################################################

# Returning the predictor Column categories


@app.route('/columnsNames/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def columnsNames():
    global X
    global data_frame1
    # Names of all the columns of the file
    columns = list(data_frame1.columns.values)
    # data_frame3 = data_frame1.drop( axis=0)
    # print("\n\n data_frame3: \n\n\n", data_frame1(axis=0) ) 
    data_frame2 = data_frame1.values.tolist()
    data_frame2 = jsonpify(data_frame2)

    # return data_frame2
    return jsonify(columns)
    # return "column names"

# #########################################################################################################################

# Returning the predicted values of the column selected


@app.route('/predictedColumn/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def predictedColumn():
    print(request)
    predicterColumn = request.form['predicterColumn']

    return "column names"

# #########################################################################################################################

# Returning the predicted values of the column selected


@app.route('/firstColumn/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def firstColumn():
    global X
    global data_frame
    X2 = data_frame1.iloc[:,0]
    print (data_frame1)
    X2 = X2.values.tolist()
    # y2 = jsonpify(y2)
    print("\n\n X2 \n", type(X2))
    return jsonify(X2)

    return "column names"

# #########################################################################################################################

# Returning the predicted values of the column selected


@app.route('/predictedFile/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def predictedFile():
    global X_train
    global X_test
    global y_train
    global y_test
    global algorithm
    global linear
    global LR_clf
    global DT_clf
    global knn_clf
    global svc_clf
    global nbclf
    global RF_clf
    global data_frame1
    global y_pred
    print("\n\n\n y_pred \n\n\n\n",y_pred)
    # print("\n\n\n y_test \n\n\n\n",y_test)
    print("\n\n\n data_frame \n\n\n\n",data_frame)

    y_pred2 = y_pred.values.tolist()
    # y_pred2 = jsonpify(y_pred2)        
        
    print("\n\n y_pred2 \n", type(y_pred2))
    return jsonify(y_pred2)


# ##########################################################################################################################

# Returning the required visualization data


@app.route('/dataFileDetails/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def dataFileDetails():
    print('\n\nrequest\n\n')
    print(request)
    col_name1 = request.form['name1']
    col_name2 = request.form['name2']
    print(col_name1)
    print(col_name2)
    # col_name = request.data
    # print(request.data)
    # col_name1 = col_name1.decode('utf-8')
    # col_name2 = col_name2.decode('utf-8')
    print('Calling the uplaod file2: ')
    global data_frame
    # global X_data
    # X_data = data_frame1.iloc[:, :-1]
    # y_data = data_frame1.iloc[:, [-1]]
    # X_data = X_data.to_json()
    # y_data = y_data.to_json()
    col_categories = data_frame[col_name1].value_counts()
    # particular column is the column that is to be predicted
    particular_column = data_frame[col_name1]
    particular_column = particular_column.to_json()
    y_data = data_frame[col_name2]
    y_data = y_data.to_json()
    col_categories = col_categories.to_string()
    return jsonify(col_categories, particular_column, y_data)




if __name__ == "__main__":
    app.run(debug=True, port=5000)
