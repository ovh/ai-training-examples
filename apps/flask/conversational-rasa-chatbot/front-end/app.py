from flask import Flask
from flask import render_template
import jwt
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)


@app.route("/")
def hello_world():
    encoded_jwt = jwt.encode({"user": "flask-app"},
                             str(os.getenv('JWT_SECRET_KEY')), algorithm="HS256")
    # Create the url back end to have the rasa model train.
    # print(str(os.getenv('JWT_SECRET_KEY')))
    api_url = os.getenv("API_URL")
    if (api_url == None):
        api_url = 'http://localhost:5005/'
    return render_template("chatbot.html", jwebtoken=encoded_jwt, url=api_url)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
