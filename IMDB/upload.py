import requests
import json

def upload_for_eval(url, model_file_path):
    with open(model_file_path, "rb") as model_file:
        files = {"model": model_file}
        response = requests.post(url, files=files)
    
    # Pretty print the response from the server
    print(json.dumps(response.json(), indent=4))
