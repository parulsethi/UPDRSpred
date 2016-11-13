import urllib.request
import json

with urllib.request.urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data') as response:
    html = response.readlines()

data = []

for _ in html:
    row = _.decode('UTF-8').replace('\n','').split(',')
    data.append(row)

json.dump(data,open('data.json', 'w'),indent = 4)
