import tweepy
from datetime import datetime
import csv
import os
import re
import json
# import daemon
from config import getConfig

def test(api):
    # Test api
    public_tweets = api.home_timeline()
    print("Test home timeline:", len(public_tweets))


def load_config_tk(filename):
    with open(filename, "r") as f:
        consumer_key, consumer_secret, access_token, access_token_secret = [line.strip() for line in f.readlines()]
        print(consumer_key, consumer_secret, access_token, access_token_secret)
        return consumer_key, consumer_secret, access_token, access_token_secret

def loadFilter(filename):
    with open(filename, "r") as filterJson:
        j = json.load(filterJson)
        return j
    
def sperateBoundbox(degree):
    minlong = boundbox[0]
    minlat = boundbox[1]
    maxlong = boundbox[2]
    maxlat = boundbox[3]

    degree = 1000

    boxs = []
    detal_lat = (maxlat - minlat) / degree
    detal_long = (maxlong - minlong) / degree

    l1 = minlong
    l2 = minlat

    for x in range(degree):
        l1 = minlong
        l2 += detal_lat
        l4 = l2 + detal_lat
        for y in range(degree):
            l1 += detal_long
            l3 = l1 + detal_long
            box = [l1, l2, l3, l4]
            boxs.append(box)
    return boxs


def create_csv(filename):
    print("Start running on", filename)
    with open(filename, "w") as outfile:
        writer = csv.writer(outfile)
        if getAllTweets:
            writer.writerow(title)
        else:
            writer.writerow(jsonKeys)


def write_csv(data):
    filename = datetime.now().strftime("%Y%m%d-%H") + ".csv"
    if os.path.isfile(filename):
        pass
    else:
        create_csv(filename)
    with open(filename, "a") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)

def removeStopWord(text):
    return ' '.join([word for word in text.split() if word not in tracks])
        
def cursorGenerator(api):
    filename = datetime.now().strftime("%Y%m%d-%H-%M") + ".json"
    if os.path.isfile(filename):
        print("{} existing, return".format(filename))
        return
    with open(ouputFolder + filename, 'w') as outfile:
        tweetJsonArray = []
        for status in tweepy.Cursor(api.search, q=search_term, since = '2020-03-01').items(10000):
            # process status here
            text = removeStopWord(status.text)
            if len(text) > 1:               
                # only essentail data 
                print(text)
                tweetJsonArray.append(status._json)
                break
        json.dump(IteratorAsList(tweetJsonArray), outfile)
        

class IteratorAsList(list):
    def __init__(self, it):
        self.it = it
    def __iter__(self):
        return self.it
    def __len__(self):
        return 1
            
########
# setup config
ouputFolder = "./twitter-data-covid/"

tokenPath = '../token.tk'
consumer_key, consumer_secret, access_token, access_token_secret = load_config_tk(tokenPath)

config = getConfig()
title = config['title']
jsonKeys = config['jsonKeys']
tracks = config['track']
# search_term = "{} -filter:retweets".format(" OR ".join(tracks))
search_term = "{}".format(" OR ".join(tracks))

#setup boundbox
filters = loadFilter("./filter.json")
#setup geoconfig: if Ture, get every tweet; if False, only tweet with geotag will write down
ignoreGeoTag = False
# setup if all the twitter data is needed, not recommend cuz it may cause lots of unnecessary data
getAllTweets = False
########

def main():
    # setup api
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    
    # test
    test(api)
    
    # with daemon.DaemonContext():
    print("Start running on", datetime.now().strftime("%Y%m%d-%H%M%S"))
    cursorGenerator(api)



if __name__ == "__main__":
    print("Start running...")
    main()