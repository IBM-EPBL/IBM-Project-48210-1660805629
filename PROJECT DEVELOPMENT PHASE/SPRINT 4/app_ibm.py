

import flask
from flask import request, render_template
from flask_cors import CORS
import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "9dafixPBcRly9wS1KiSmQf5Ed3irsZnLXW6Jnfuj6MDX"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}


app = flask.Flask(__name__, static_url_path='')
CORS(app)
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/neww')
def neww():
    return render_template("indexnew.html")
@app.route('/last',methods=['POST','GET'])
def last():
    hb = float(request.form['hb'])
    sg = float(request.form['sg'])
    rbcc = float(request.form['rbcc'])
    ab = float(request.form['ab'])
    bu = float(request.form['bu'])
    bp = float(request.form['bp'])
    bgr = float(request.form['bgr'])
    sc = float(request.form['sc'])
    final_features = [[hb,sg,rbcc,ab,bu,bp,bgr,sc]]
   
    payload_scoring = {"input_data": [{"field": [['hb','sg','rbcc','ab','bu','bp','bgr','sc']], "values":   final_features}]}
    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/8ad8a3ba-e143-41f6-ac61-01a7f3b24547/predictions?version=2022-11-09', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})
    print(response_scoring)
    predictions = response_scoring.json()
    predict = predictions['predictions'][0]['values'][0][0]
    
    if predict==0:
            return render_template('neg.html')
    else:
        return render_template('pos.html')

if __name__ == '__main__' :
    app.run(debug= False)
