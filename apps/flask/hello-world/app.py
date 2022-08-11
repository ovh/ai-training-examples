from flask import Flask

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return 'Web App with Python Flask using AI Training!'

if __name__ == '__main__':
    # starting app
    app.run(debug=True,host='0.0.0.0')
