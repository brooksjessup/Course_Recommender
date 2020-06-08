# imports
import numpy as np
import pickle
from flask import Flask, request, Response, render_template, jsonify
import gensim
import pandas as pd

# initialize the flask app
app = Flask('cr_app')

@app.route('/')
def home():
    return render_template('form.html')

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
    top = 10 # Set how many courses to recommend
    sims = model.docvecs.most_similar([vector], topn=top) # Find the most similar vectors
    course_ids = [sim[0] for sim in sims] # Extract the course index numbers
    df = pd.read_csv('./Data/Course_Data/Coursera_Catalog.csv') # Read in course data
    course_names = [df.iloc[id]['name'] for id in course_ids] # Get the course names
    return render_template('results.html', recommendations=course_names, len=top)

# Call app.run(debug=True) when python script is called
if __name__ == '__main__': # if you run 'python app_starter.py' from terminal
    app.run(debug=True)
