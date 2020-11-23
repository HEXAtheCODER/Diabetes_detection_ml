from flask import Flask,render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model")


@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/result',methods=['POST'])
def predict():
    list_of_features = [float(x) for x in request.form.values()]
    final_features = [np.array(list_of_features)]
    prediction = model.predict(final_features)


    return render_template("result.html",result = prediction[0])


app.run(debug=True)
