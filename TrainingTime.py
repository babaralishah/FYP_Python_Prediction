#Import Libraries :

import pandas as pd
from flask import jsonify
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#import various ML algorithms to be used from the library
from sklearn.preprocessing import MinMaxScaler

#Naive Bayes Classifier
from sklearn import naive_bayes

#SVM
from sklearn.svm import SVC,NuSVC

#LabelEncoder
from sklearn.preprocessing import LabelEncoder

#KNN
from sklearn.neighbors import KNeighborsClassifier

#Splitting
from sklearn.model_selection import train_test_split

#NB
from sklearn.naive_bayes import GaussianNB,MultinomialNB

#Logistic Regression
from sklearn.linear_model import SGDClassifier, LogisticRegression

#DTClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

#RFC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import csv
from flask import make_response, url_for,Blueprint
from werkzeug.utils import secure_filename
import pandas as pad
from flask_cors import CORS, cross_origin
import numpy as np
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, redirect
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics
TrainingTime = Blueprint("TrainingTime",__name__)

CORS(TrainingTime, supports_credentials=True)

@TrainingTime.route('/trainingTime/', methods = ['POST','GET'])
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
    print("TrainingTime area")
    return jsonify([trainingTime])
