import requests
import numpy as np

url = "http://0.0.0.0:8000/"
response = requests.get(url=url)
response.status_code
response.text

url = "http://0.0.0.0:8000/predict"

payload = {
    "name": "Sandeep",
    "score": 79.8,
    "id": 23
}

response = requests.post(url=url, json= payload)
response.status_code
response.text


response.text

