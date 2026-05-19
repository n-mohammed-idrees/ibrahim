from flask import Flask,request,jsonify
import numpy as np
import joblib
app=Flask(__name__)
model=joblib.load("diabetes_model.pkl")
scaler=joblib.load("diabetes_scaler.pkl")
@app.route("/predict",methods=["POST"])
def predict():
    x=request.json
    x=np.asanyarray(x)
    x=x.reshape(1,-1)
    x=scaler.transform(x)
    a=model.predict(x)
    print(a)
    return jsonify({"ppp":int(a)})
if __name__=="__main__":    
    app.run(host="0.0.0.0",port=10000)