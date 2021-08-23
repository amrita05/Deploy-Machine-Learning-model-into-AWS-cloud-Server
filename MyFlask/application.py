# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 15:16:35 2021

@author: AMRITA GAUTAM
"""

from flask import Flask, request
import joblib
application=Flask(__name__)

vectorizer=joblib.load('vectorizer.pkl')
spamorham_model=joblib.load('spam_ham_model.pkl')

sentiment_model=joblib.load('Sentiment_analysis.pkl')


@application.route('/')
def hello_world():
    return "Hello World"

@application.route('/spamorham',methods=['GET','POST'])
def spamorham():
    message=request.args.get("message")
    vect_msg=vectorizer.transform([message])
    result = spamorham_model.predict(vect_msg)[0]    
    return result

@application.route('/sentiment',methods=['GET','POST'])
def sentiment():
    message=request.args.get("message")
    result = spamorham_model.predict([message])[0]    
    return result

if __name__=='__main__':
    application.run()
    
