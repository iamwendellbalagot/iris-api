import requests

res = requests.post(
    url='http://localhost:5000/',
    json={
        'sepal_length': 3.0,
        'sepal_width': 4.2,
        'petal_length': 0.5,
        'petal_width': 3.3
    }
)

print(res.json())