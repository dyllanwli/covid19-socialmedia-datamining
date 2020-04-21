import nltk
import json
import re, json, string, datetime, random, itertools

hashtag_df_path = "/scratch/user/diya.li/kaggle-data/covid-twitter/Hashtags.CSV"
keywords_path = "/scratch/user/diya.li/twitter-action/depression/keywords.txt"
keywords2_path = "/scratch/user/diya.li/twitter-action/depression/keywords2.txt"
stopword_path = "/scratch/user/diya.li/twitter-action/depression/long_stop_words.json"


def read_keyword2(path):
    keyword_set = []
    with open(path, "r") as f:
        line = f.readline()
    return set(eval(line))


def read_keyword(path):
    keyword_set = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            line = "".join([x for x in line.split(" ")[:-1] if len(x) > 0])
            line = line.lower()
            keyword_set.append(line)
    keyword_set = set(keyword_set)
    return keyword_set


def clean_text_with_keyword(keyword_set, text):
    text = str(text).lower()
    return " ".join([word for word in text.split() if word not in keyword_set])


def read_n_clean_df(input_df_path):
    # use keyword to remove all related text
    keyword_set = read_keyword2(keywords2_path)

    for each_input_df_path in input_df_path:
        df_temp = pd.read_csv(each_input_df_path)
        print("Reading", each_input_df_path, df_temp.shape)
        df_temp["cleaned_text"] = df_temp["text"].apply(
            lambda x: clean_text_with_keyword(keyword_set, x)
        )
        df_temp.to_csv(each_input_df_path, index=False)
    print("done")
    

def read_stopwords():
    # read long stop words

    punctuation = list(string.punctuation)
    punctuation.remove("-")
    punctuation.remove("_")

    with open(stopword_path, "r") as f:
        long_stop_list = json.load(f)

    # get full stop words for this case
    stopwords = nltk.corpus.stopwords.words("english")
    stoplist = long_stop_list + punctuation
    stopwords.extend(stoplist)
    stopwords.extend(["al", "mon", "vis"])
    len(stopwords)
    return stopwords