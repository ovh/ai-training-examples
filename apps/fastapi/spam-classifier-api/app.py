# import dependencies
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from model import model_logistic_regression, index, vectorizer

# initialize an instance of fastapi
app = FastAPI()

# define the data format
class request_body(BaseModel):
    message : str

# process the message sent by the user
def process_message(message):

    # remove stop words and transform the message for prediction
    desc = vectorizer.transform(message)
    dense_desc = desc.toarray()
    dense_select = dense_desc[:, index[0]]

    # return the processed message
    return dense_select

# GET method
@app.get('/')
def root():
    return {'message': 'Welcome to the SPAM classifier API'}

# POST method
@app.post('/spam_detection_path')
def classify_message(data : request_body):

    # message formatting
    message = [
        data.message
    ]

    # check if the message exists
    if (not (message)):
        raise HTTPException(status_code=400, detail="Please Provide a valid text message")

    # process the message to fit with the model
    dense_select = process_message(message)

    # classification results
    label = model_logistic_regression.predict(dense_select)
    proba = model_logistic_regression.predict_proba(dense_select)

    # extract the corresponding proba
    if label[0]=='ham':
        label_proba = proba[0][0]
    else:
        label_proba = proba[0][1]

    # return the results!
    return {'label': label[0], 'label_probability': label_proba}
