# Disaster_Response_Pipelines


1. ETL Pipeline
In a Python script, process_data.py,
(We expect you to do the data cleaning with pandas. To load the data into an
SQLite database, you can use the pandas dataframe .to_sql() method, which you
can use with an SQLAlchemy engine.)
write a data cleaning pipeline that:
  - Loads the messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database

python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db


2. ML Pipeline
In a Python script, train_classifier.py,
(you will split the data into a training set and a test set. Then, you will
create a machine learning pipeline that uses NLTK, as well as scikit-learn's
Pipeline and GridSearchCV to output a final model that uses the message column
to predict classifications for 36 categories. Finally, you will export your
model to a pickle file)
write a machine learning pipeline that:
  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file

python train_classifier.py ../data/DisasterResponse.db classifier.pkl

3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra
features depending on your knowledge of flask, html, css and javascript.
For this part, you'll need to:
  - Modify file paths for database and model as needed
  - Add data visualizations using Plotly in the web app. One example is provided for you
  - Github and Code Quality

python run.py
env|grep WORK
https://SPACEID-3001.SPACEDOMAIN
https://viewa7a4999b-3001.udacity-student-workspaces.com

Your project will also be graded based on the following:
  - Use of Git and Github
  - Strong documentation
  - Clean and modular code
  - Follow the RUBRIC when you work on your project to assure you meet all of the necessary criteria for developing the pipelines and web app.
