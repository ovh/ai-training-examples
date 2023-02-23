# import objects from the Flask model
from flask import Flask, jsonify, render_template, request, make_response
import transformers

# creating flask app
app = Flask(__name__)

# create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
dictOfModels = {"BERT" : transformers.pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")} # feel free to add several models

listOfKeys = []
for key in dictOfModels :
        listOfKeys.append(key) 

# inference fonction
def get_prediction(message,model):
    # inference
    results = model(message)  
    return results

# get method
@app.route('/', methods=['GET'])
def get():
    # in the select we will have each key of the list in option
    return render_template("home.html", len = len(listOfKeys), listOfKeys = listOfKeys)


# post method
@app.route('/', methods=['POST'])
def predict():
    message = request.form['message']
    
    # choice of the model
    results = get_prediction(message, dictOfModels[request.form.get("model_choice")])
    print(f'User selected model : {request.form.get("model_choice")}')
    my_prediction = f'The feeling of this text is {results[0]["label"]} with probability of {results[0]["score"]*100}%.'
 
    return render_template('result.html', text = f'{message}', prediction = my_prediction)

if __name__ == '__main__':
    # starting app
    app.run(debug=True,host='0.0.0.0')
