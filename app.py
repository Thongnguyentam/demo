from logging import debug
from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods = ['GET']) #sends data in unencrypted form to server
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_feature = [float(x) for x in request.form.values()]
    final = np.reshape(int_feature, (6,1))
    prediction = model.feedforward(final)*100
    output='{0:.{1}f}'.format(prediction[1][0], 2)

    if output>str(50):
        return render_template('predict.html',pred='You are likely to get exposed to Covid-19.\nProbability of getting Covid-19 is {} %'.format(output))
    else:
        return render_template('predict.html',pred='You are not likely to get exposed to Covid-19.\n Probability of getting Covid-19 is {} %'.format(output))

@app.route('/predictco', methods = ['GET'])
def predictco():
    return render_template('predict.html')

if __name__ == "__main__":
    por = os.environ.get("PORT", 5000) #Heroku will set the PORT environment variable for web traffic
    app.run(debug = False, host = "0.0.0.0", port = por)