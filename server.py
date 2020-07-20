# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np


classifier = pickle.load(open('diabetes.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        prediction = classifier.predict(data)

        if prediction == 1:
            pred = "You have Diabetes, please consult a Doctor."
        elif prediction == 0:
            pred = "You don't have Diabetes."
        output = pred

        return render_template('index.html', prediction_text='{}'.format(output))





if __name__ == '__main__':
    app.run(debug=True)