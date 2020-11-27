from flask import Flask,render_template, request, send_from_directory
import numpy as np
import joblib
import os


app = Flask(__name__)
model = joblib.load("model")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/detect')
def detect():
    return render_template("detect.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/result', methods=['POST'])
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
# if __name__ == "__main__":
#     app.run(host='0.0.0.0',debug=True)
