import os
import json
import joblib
import pandas as pd  # pandas is not preinstalled in sklearn framework container, so we include it in requirements.txt, and import here as an example
import numpy as np  # numpy is preinstalled in sklearn framework container, so import here directly as an example

# Model serving
"""
Deserialize fitted model
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
I leave a piece of sample code here in case your request_content_type is not "text/csv" or you want to do some preprocessing
"""
# def input_fn(request_body, request_content_type="application/json"):
#     if request_content_type == "application/json":
#         request_body = json.loads(request_body)
#         inpVar = request_body["Input"]
#         return inpVar
#     else:
#         raise ValueError("This model only supports application/json input")

"""
predict_fn
    input_data: returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""
def predict_fn(input_data, model):
    return model.predict(input_data)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string
I also leave a piece of sample code here in case you want to do any postprocessing of the prediction, eg, change the output format
"""
# def output_fn(prediction, content_type):
#     res = int(prediction[0])
#     respJSON = {"Output": res}
#     return respJSON
