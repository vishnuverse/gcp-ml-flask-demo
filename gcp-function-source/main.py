import requests
import pickle
from google.cloud import storage
import numpy as np


def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    request_json = request.get_json()
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('ml-flask-2021')
    blob_classifier = bucket.blob('models/classifier.pickle')
    blob_scaler = bucket.blob('models/sc.pickle')
    blob_classifier.download_to_filename('/tmp/classifier.pickle')
    blob_scaler.download_to_filename('/tmp/sc.pickle')
    # serverless_classifier = pickle.load(open('/tmp/classifier.pkl','rb'))
    with open("/tmp/classifier.pickle", "rb") as f:
        serverless_classifier = pickle.load(f)
    # serverless_scaler = pickle.load(open('/tmp/sc.pkl','rb'))
    with open("/tmp/sc.pickle", "rb") as f:
        serverless_scaler = pickle.load(f)
    age = request_json['age']
    salary = request_json['salary']
    pred_proba = serverless_classifier.predict_proba(serverless_scaler.transform(np.array([[age,salary]])))[:,1]
    print(pred_proba)
    print(age)
    print(salary)
    return "The prediction is {}".format(pred_proba)