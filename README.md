# Career Course Recommender
*A web application that uses machine learning to recommend online courses to jobseekers.*

## 1. Problem Statement
Online learning can be a valuable tool for jobseekers to acquire the skills they need to land the jobs they want, without having to leave the safety of their own homes in the middle of a pandemic. However, each individual jobseeker faces the challenge of determining which courses among the thousands available online will be the most relevant to their particular dream job. Can machine learning help them find the most relevant courses?

## 2. Data Collection
I collected course data from a leading online learning platform, Coursera. By writing a Python function to send multiple get requests to Coursera’s public [Catalog API](https://build.coursera.org/app-platform/catalog/), I gathered information for all 4,416 courses currently available on their platform. The resulting dataset includes the title, instructor, description, and other features of each of these courses.

## 3. Data Modeling
I built a machine learning model to predict which courses in the dataset would be most relevant to a given job. Using Python’s Gensim library for Natural Language Processing, I trained a Doc2Vec model to learn numerical representations for the rich textual data provided in the descriptions of the courses. The model can take any job description (such as from a job board) and calculate which course descriptions are mathematically most similar to it.

## 4. Model Evaluation
I created a custom metric to evaluate and optimize the model’s predictive performance. Because Doc2Vec is an unsupervised learning model without a target variable, there is no labeled data against which to measure the accuracy of its predictions. I addressed this problem by manually labeling a sample set of job descriptions with relevant course descriptions, and then scoring how well the model predicted their similarity. After this metric was used to tune its hyperparameters, the model is able to predict an average similarity score of about 85% for the labeled data.

## 5. Model Deployment
I developed and deployed a [web application](https://career-course-recommender.herokuapp.com/) to make the model available to jobseekers. I used the Flask web framework in combination with HTML and CSS to develop an API for receiving job descriptions from users and sending course recommendations in return. I then deployed the app to production on the Heroku platform where it is publicly accessible to jobseekers.
