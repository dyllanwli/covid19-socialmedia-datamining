import pandas as pd
import numpy as np
import os
import ujson
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm


class PreprocessTwitter:
    def __init__(self):
        self.base_path = os.environ["SCRATCH"]

        input_folder = "covid-map/twitter-dataset-covid-all"
        self.input_folder_path = os.path.join(self.base_path, input_folder)

        output_folder = "covid-map/twitter-dataset-processed-1"
        self.output_folder_path = os.path.join(self.base_path, output_folder)

        # csv path list splited by month
        self.tweets_filepath_set = self._read_dirs(self.input_folder_path)
        self.sample_json_file_path = self.tweets_filepath_set["2020-01"][0]

        self.tweet_columns = ["created_at", "id", "full_text", "cleaned_text", "entities", "retweet_count", "favorite_count", "CountyId",
                              "user_name", "user_followers_count", "user_friends_count", "user_listed_count", "favourites_count", "user_location", "geo"]

        self.output_file_path = os.path.join(self.output_folder_path, "concated_df.csv")

    def _read_dirs(self, input_path):
        tweets_file_set = {}
        for month_folder in os.listdir(input_path):
            if month_folder.startswith("2020") and not month_folder.endswith(".zip"):
                tweets_file_set[month_folder] = []
                month_folder_path = os.path.join(input_path, month_folder)
                # print(month_folder_path)
                for tweets_file in os.listdir(month_folder_path):
                    if tweets_file.endswith("json") and tweets_file.find(")") == -1:
                        # some file is duplicated
                        tweets_file_path = os.path.join(
                            month_folder_path, tweets_file)
                        tweets_file_set[month_folder].append(tweets_file_path)

        print("filepath:", tweets_file_set.keys())
        # print("all file count", sum([len(tweets_file_set[x]) for x in tweets_file_set]))
        return tweets_file_set

    def _clean_text(self, text):
        # Check characters to see if they are in punctuation
        nopunc = [char for char in text if char not in string.punctuation]
        # Join the characters again to form the string
        nopunc = "".join(nopunc)
        # convert text to lower-case
        nopunc = nopunc.lower()
        # remove URLs
        nopunc = re.sub(
            "((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))", "", nopunc
        )
        nopunc = re.sub(r"http\S+", "", nopunc)
        # remove usernames
        nopunc = re.sub("@[^\s]+", "", nopunc)
        # remove the # in #hashtag
        nopunc = re.sub(r"#([^\s]+)", r"\1", nopunc)
        # remove numbers
        nopunc = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", nopunc)
        nopunc = re.sub("\d", "", nopunc)
        # remove repeated characters
        nopunc = re.sub("(RT|corona|covid|virus)", "", nopunc)
        nopunc = word_tokenize(nopunc)
        # remove stopwords from final word list
        nopunc = [
            word for word in nopunc if word not in stopwords.words("english")]
        text = " ".join([str(elem) for elem in nopunc]) + "\n"
        return text
    
    def write_to_csv(self, tweet_list, output_path):
        df = pd.DataFrame(tweet_list, columns=self.tweet_columns)
        df.to_csv(output_path, index=False)

    def tweets_filter(self, json_obj_array):
        '''
        input json
        return row list
        '''
        row_list = []
        for json_obj in json_obj_array:
            created_at = pd.to_datetime(json_obj.get("created_at")) 
            # concverted to pd datatime
            status_id = json_obj.get("id")
            full_text = json_obj.get("full_text")
            cleaned_text = self._clean_text(full_text)
            # clean text
            entities = json_obj.get("entities")
            retweet_count = json_obj.get("retweet_count")
            favorite_count = json_obj.get("favorite_count")
            CountyId = json_obj.get("CountyId")
            if CountyId is None:
                continue
            user = json_obj.get("user")
            user_name = user.get("name")
            user_followers_count = user.get("followers_count")
            user_friends_count = user.get("friends_count")
            user_listed_count = user.get("listed_count")
            user_favourites_count = user.get("favourites_count")
            user_location = user.get("location")
            geo = json_obj.get("geo")

            row = [created_at, status_id, full_text, cleaned_text, entities, retweet_count, favorite_count, CountyId,
                   user_name, user_followers_count, user_friends_count, user_listed_count, favourites_count, user_location, geo]
            row_list.append(row)

        return row_list

    def read_one_json(self, json_path):
        '''
        return row list
        '''
        with open(json_path, "r") as j:
            json_obj = ujson.load(j)
            row_list = self.tweets_filter(json_obj_array)
            if row_list is not None:
                return row_list
            else:
                return None

    def read_all_json(self, json_path_list):
        nums = len(json_path_list)
        print("reading count", nums)
        tweet_list = []
        with tqdm(total=nums) as pbar:
            for json_path in json_path_list:
                row_list = self.read_one_json(json_path)
                tweet_list.extend(row_list)
                break
                pbar.update(1)
            print("all done")
        return tweet_list
    
    def start_all(self, json_path_list):
        tweet_list = self.read_all_json(json_path_list)
        self.write_to_csv(tweet_list, self.output_file_path)