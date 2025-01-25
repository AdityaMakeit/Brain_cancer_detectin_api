import requests

# Example of making a GET request to the Flask API
url = 'http://127.0.0.1:5000/predict'
params = {'img_path': r'C:\Users\Aditya\Desktop\code\Brain tumor\test\Tumor\Y183.jpg'}

response = requests.get(url, params=params)
print(response.json())
