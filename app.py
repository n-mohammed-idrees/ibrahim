from flask import Flask,request,jsonify
app=Flask(__name__)
@app.route("/predict",methods=["POST"])
def predict():
    try:
        import numpy as np
        import joblib
        model=joblib.load("diabetes_model.pkl")
        scaler=joblib.load("diabetes_scaler.pkl")
        x=request.json
        print(x)
        x=np.asanyarray(x)
        x=x.reshape(1,-1)
        x=scaler.transform(x)
        a=model.predict(x)
        print(a)
        return jsonify({"ppp":int(a)})
    except Exception as e:
        return jsonify({"error",str(e)})