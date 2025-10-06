import requests

API_URL = 'http://127.0.0.1:8000/predict'

sample = {
    'model_name': 'model_latest.pkl',
    'data': [
        {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2},
    ]
}

r = requests.post(API_URL, json=sample)
print(r.status_code)
print(r.json())
