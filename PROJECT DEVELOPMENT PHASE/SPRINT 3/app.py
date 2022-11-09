
import flask
import numpy as np
import pickle

from flask import request, render_template,Flask


from flask_cors import CORS

import joblib
 
app = Flask(__name__, static_url_path='')
CORS(app)
model = pickle.load(open('ckd.pkl', 'rb'))

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
    prediction = model.predict(final_features)
       
    if prediction==0:
        return render_template('neg.html')
    else:
        return render_template('pos.html')


   
    
        

if __name__ == '__main__':
    app.run(debug=True)
    
        
        
    

 

    
            
        
            
       
        
       
  

       

 

        

