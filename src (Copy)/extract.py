import requests

url = "https://data.cityofchicago.org/resource/wrvz-psew.json"

params = {
    "$limit": 10
}

response = requests.get(url, params=params)

response.raise_for_status()

data = response.json()

print(len(data))
print(data[0])