# imports
import numpy as np
import pickle
from flask import Flask, request, Response, render_template, jsonify
import gensim

# initialize the flask app
app = Flask('cr_app')

# route 1: hello world
@app.route('/')
def home():
    # return a simple string
    return '<html><body><h1>Welcome to Course Recommender!</h1></body></html>'

# route 2: show a form to the user
@app.route('/form')
def form():
    # use flask's render_template function to display an html page
    return render_template('form.html')


# route 3: accept the form submission and do something fancy with it
@app.route('/submit')
def submit():
    js = request.args["JobDesc"] # Load in the form data
    doc = gensim.utils.simple_preprocess(js) # Preprocess the job description
    model = pickle.load(open('./model.p', 'rb')) # Load the model
    vector = model.infer_vector(doc) # Vectorize the job description
    sims = model.docvecs.most_similar([vector])
    return render_template('results.html', recommendation=sims)

# Call app.run(debug=True) when python script is called
if __name__ == '__main__': # if you run 'python app_starter.py' from terminal
    app.run(debug=True)
