from logging import debug
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods = ['GET']) #sends data in unencrypted form to server
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_feature = [int(x) for x in request.form.values()]
    final = np.array(int_feature)
    final.transpose()
    prediction = model.feedforward(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output>str(0.5):
        return render_template('predict.html',pred='You are likely to get exposed to Covid-19.\nProbability of getting Covid-19 is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('predict.html',pred='You are not likely to get exposed to Covid-19.\n Probability of getting Covid-19 is {}'.format(output),bhai="Your Forest is Safe for now")

@app.route('/predictco', methods = ['GET'])
def predictco():
    return render_template('predict.html')

if __name__ == "__main__":
    app.run(port=3000, debug = True)