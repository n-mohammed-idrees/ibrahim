from flask import Flask,request,jsonify
import numpy as np
import pandas as pd
import joblib
app=Flask(__name__)
model=joblib.load(r"C:\Users\nmoha\OneDrive\Desktop\ibrahim-projects\Backend\diabetes_model.pkl")
scaler=joblib.load(r"C:\Users\nmoha\OneDrive\Desktop\ibrahim-projects\Backend\diabetes_scaler.pkl")
@app.route("/predict",methods=["POST"])
def predict():
    x=request.json
    x=np.asanyarray(x)
    x=x.reshape(1,-1)
    x=scaler.transform(x)
    a=model.predict(x)
    print(a)
    return jsonify({"ppp":int(a)})
    
app.run(debug=True)