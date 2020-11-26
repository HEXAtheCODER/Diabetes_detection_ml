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

# IMPORTANT
# The following line, app.run(debug=True) should only be uncommented when running the app locally using 'flask run'.
# On heroku, the app is started by the command 'web: gunicorn flask_app:app'
# If 'app.run()' is used, then the app will crash with error 'Address already in use', since gunicorn is already running
# DO NOT UNCOMMENT THE FOLLOWING LINE WHEN PUSHING TO GIT

# app.run(debug=True)
