# NapaAI

Contributors: Siddharth Vasu, Aravind Reddy Aloori, Jackson Cross, Tyler Nguyen

# About

Recommender AI for wines that takes into account title, variety, description, and rating to make novel and relevant recommendations.

# Build Instructions
Python 3 is required to run this web app.

To run, navigate to the project directory, and enter into the terminal:
`./run.sh`

Then click on the link to localhost.

# Layout

The data is stored in the data directory. It is used by the model.py file to train the model, and the model is then saved under wine_model so that it can be simply reloaded every time you run the app. Then the test.py file contains functions to utilize the model, include the recommend function which returns recommendations. When the run.sh file is executed, it installs the dependencies listed in the requirements.txt file, and then runs the Flask app located in the app.py file. The app loads in HTML/CSS files located in the templates and static folders to allow users to navigate the app, as well as calling functions from the test.py file to make recommendations.
