import numpy as np
import pickle
import pandas
import os
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = pickle.load(open('C:/Users/Pc/Desktop/Mental_Health_1/Mental_Boosting.pkl','rb'))
     
@app.route('/')
def home():
    return render_template('index.html')       

@app.route('/predict', methods = ["POST", "GET"]) 
def predict():
    input_feature = [int(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]

    feature_cols = [['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']]    

    data = pandas.DataFrame(features_values, columns = feature_cols)

    prediction = model.predict(data)
    print(prediction)
                                                                                       
    if(prediction == 0):
        return render_template("mental_result.html", prediction_text = "Treatment is not required")
    else:
        return render_template("mental_result.html", prediction_text = "Treatment is required")

@app.route('/index.html')
def mental_result():
    return render_template('mental_result.html')    

if __name__=="__main__":
    port = int(os.environ.get('PORT',5000))
    app.run(port = port, debug = True, use_reloader = False)