import json

def getConfig():
    # track data source: https://trendogate.com/
    with open('./cursor-config.json', 'r') as c:
        return json.load(c)
