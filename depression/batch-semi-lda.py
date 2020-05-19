import pandas as pd
import numpy as np
import nltk
from scipy.sparse import coo_matrix
import re, json, string, datetime, random, itertools

from collections import OrderedDict, defaultdict

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import os
from corextopic import corextopic as ct
import scipy.sparse as ss
from corextopic import (
    vis_topic as vt,
)  # jupyter notebooks will complain matplotlib is being loaded twice

from difflib import get_close_matches
# got issues from https://stackoverflow.com/questions/26283715/how-to-find-the-most-similar-word-in-a-list-in-python


def read_stopwords():
    # read long stop words
    punctuation = list(string.punctuation)
    punctuation.remove("-")
    punctuation.remove("_")
    with open("long_stop_words.json", "r") as f:
        long_stop_list = json.load(f)
    # get full stop words for this case
    stopwords = nltk.corpus.stopwords.words("english")
    stoplist = long_stop_list + punctuation
    stopwords.extend(stoplist)
    stopwords.extend(["al", "mon", "vis"])
    len(stopwords)
    return stopwords


def read_seed_list():
    # read depression lexicon
    with open("depression_lexicon-phq-9-new.json") as f:
        seed_terms = json.load(f)
    all_seeds_raw = [
        seed.replace("_", " ")
        for seed in list(
            itertools.chain.from_iterable(
                [seed_terms[signal] for signal in seed_terms.keys()]
            )
        )
    ]
    seed_lists = [
        [item.replace("_", " ").lower() for item in seed_terms[k]]
        for k in seed_terms.keys()
    ]
    return seed_lists


def load_pretrained_glove():
    scratch_path = os.environ["SCRATCH"]
    filename = "glove.6B.50d.txt"
    glove_path = os.path.join(scratch_path, "tmp/glove6B")

    embeddings_dict = {}
    with open(os.path.join(glove_path, filename), "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            token = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[token] = vector
    return embeddings_dict


class BatchProcessSemiLDA:
    def __init__(self):

        # path config
        self.scratch_path = os.environ["SCRATCH"]
        self.input_folder = "covid-map/twitter-dataset-processed-stress-2W"
        self.input_path = os.path.join(self.scratch_path, self.input_folder)

        self.input_df_path = [
            os.path.join(self.input_path, x)
            for x in os.listdir(self.input_path)
            if x.endswith(".csv")
        ]

        # csv path list splited by month
        # self.tweets_filepath_set = self._read_dirs(self.input_path)
        # sample for testing
        # self.sample_json_path = self.tweets_filepath_set["2020-01"][0]

        #         self.output_folder = "twitter-action/depression/2D-windows-stress-topic"
        self.output_folder = "covid-map/twitter-dataset-processed-topic-2W"
        self.output_folder_path = os.path.join(self.scratch_path, self.output_folder)

        self.stress_rate_threshold = 0

        # corpus
        self.stopwords = read_stopwords()
        self.seed_lists = read_seed_list()
        # df_sample temp
        self.df_sample = pd.DataFrame()
        self.anchors = None
        # used to fit word vector
        self.embeddings_dict = load_pretrained_glove()
        # topic columns
        self.topic_columns = ["topic_" + str(x) for x in range(9)]

    def _read_dirs(self, input_path):
        tweets_file_set = {}
        for month_folder in os.listdir(input_path):
            if month_folder.startswith("2020") and not month_folder.endswith(".zip"):
                tweets_file_set[month_folder] = []
                month_folder_path = os.path.join(input_path, month_folder)
                # print(month_folder_path)
                for tweets_file in os.listdir(month_folder_path):
                    if tweets_file.endswith("csv"):
                        tweets_file_path = os.path.join(month_folder_path, tweets_file)
                        tweets_file_set[month_folder].append(tweets_file_path)

        print("filepath:", tweets_file_set.keys())
        # print("all file count", sum([len(tweets_file_set[x]) for x in tweets_file_set]))
        return tweets_file_set

    def _write_df_windows(self, df_sample, opp):
        df_sample.to_csv(opp, index=False)
        print("writing df done.")

    def get_vectorizer_param(self, text_list):
        vectorizer = TfidfVectorizer(
            max_df=0.3,
            min_df=10,
            max_features=None,
            ngram_range=(1, 4),
            norm=None,
            binary=True,
            use_idf=False,
            sublinear_tf=False,
            stop_words=self.stopwords,
        )

        vectorizer = vectorizer.fit(text_list)
        tfidf = vectorizer.transform(text_list)
        vocab = vectorizer.get_feature_names()
        print("vocab num", len(vocab))
        return tfidf, vocab

    def find_closest_embeddings(self, embedding, cutoff=25):
        return sorted(
            self.embeddings_dict.keys(),
            key=lambda token: spatial.distance.euclidean(
                self, embeddings_dict[token], embedding
            ),
        )

    def depression_lexicon_pattern(self, signal):
        """deprecated function"""
        typical_signal_words = [
            [word for word in topic if (" " not in word)] for topic in self.seed_lists
        ]

        # seleect the correct target_word by index
        target_word_list = typical_signal_words[signal]

        target_embedding = self.embeddings_dict[target_word]
        for index, res in enumerate(typical_signal_words):
            if index != signal:
                # caculate all euclidean distance
                target_embedding -= self.embeddings_dict[typical_signal_words[res]]
        return target_embedding

    def get_close_matches_anchor(self, seed_lists, vocab, anchor_index):
        result = []
        seed_index = 0
        while len(result) <= 0:
            seed_word = seed_lists[anchor_index][seed_index]
            # print("seeding", seed_word)

            result = list(get_close_matches(seed_word, vocab))

            seed_index += 1
        # print("finally got one anchor", result)
        return result

    def get_anchors(self, seed_lists, vocab):
        anchor_list = []

        anchor_list = [[a for a in topic if a in vocab] for topic in seed_lists]
        anchors_len = [len(x) for x in anchor_list]
        for index, l in enumerate(anchors_len):
            if l == 0:
                print("Got an zero anchor, finding similarity anchors")
                # anyway, we have to find a word to match the voab
                anchor_list[index] = self.get_close_matches_anchor(
                    seed_lists, vocab, index
                )
        # look fine, return
        return anchor_list

    def train_model(self, X, words, anchors, anchor_strength=3):
        print("trainning model", end="\r")
        # Train the first layer
        model = ct.Corex(n_hidden=20, seed=8)
        model = model.fit(
            X,
            words=words,
            anchors=anchors,  # Pass the anchors in here
            anchor_strength=anchor_strength,  # Tell the model how much it should rely on the anchors
        )
        return model

        # TODO: Train successive layers
        tm_layer2 = ct.Corex(n_hidden=10, seed=16)
        tm_layer2.fit(model.labels)

        tm_layer3 = ct.Corex(n_hidden=9)
        tm_layer3.fit(
            tm_layer2.labels,
            words=words,
            anchors=anchors,  # Pass the anchors in here
            anchor_strength=anchor_strength,  # Tell the model how much it should rely on the anchors
            verbose=1,
            max_iter=300,
        )
        print("finished")
        return tm_layer3

    def _write_log(self, log_text_list, r_opp):
        with open(r_opp, "w") as f:
            for log in log_text_list:
                f.write("%s\n" % log)

    def print_model_topic_result(self, model, anchor_num):
        result_list = []
        for n in range(anchor_num):
            topic_words, _ = zip(*model.get_topics(topic=n))
            result = "{}: ".format(n) + ",".join(topic_words)
            result_list.append(result)
            print(result)
        return result_list

    def get_processed_df(self, model, X):
        model_labels = model.transform(X)

        # select columns https://thispointer.com/python-numpy-select-rows-columns-by-index-from-a-2d-ndarray-multi-dimension/
        model_labels = model_labels[:, : len(self.topic_columns)]

        # get topic distribution model
        topic_df = pd.DataFrame(model_labels, columns=self.topic_columns).astype(
            float
        )  # save space

        topic_df.index = self.df_sample.index
        df = pd.concat([self.df_sample, topic_df], axis=1)

        return df

    def load_n_process_df(self, sample_df_path):
        print("reading", sample_df_path)
        self.df_sample = pd.read_csv(sample_df_path, lineterminator="\n")
        # code below are processed
        # self.df_sample = self.df_sample[self.df_sample["lang"] == "en"]
        # self.df_sample = self.df_sample.drop_duplicates(subset="cleaned_text")
        # df_sample = df_sample[df_sample["cleaned_text"].notna()]
        # df_sample = df_sample[df_sample["place_type"].notna()]

        self.df_sample = self.df_sample[
            self.df_sample["stress_rate"] >= self.stress_rate_threshold
        ]
        print(self.df_sample.shape, "starting...")
        X, vocab = self.get_vectorizer_param(self.df_sample["cleaned_text"])
        self.anchors = self.get_anchors(self.seed_lists, vocab)

        if self.anchors:
            # train model
            model = self.train_model(X, vocab, self.anchors, anchor_strength=3)
            result_text_list = self.print_model_topic_result(
                model, len(self.anchors)
            )  # print result

            processed_df = self.get_processed_df(model, X)
            return processed_df, result_text_list

    def start_one(self, df_path):
        # input df: cleaned text must be cleaned
        df_file_names = df_path.split("/")[-1]
        opp = os.path.join(self.output_folder_path, df_file_names)
        r_opp = os.path.join(
            self.output_folder_path, df_file_names.replace("csv", "txt")
        )
        if os.path.isfile(opp):
            # skip exists file
            return
        processed_df, result_text_list = self.load_n_process_df(df_path)
        return processed_df, result_text_list
        print("finished one, return processed_df, result_text_list")
        #print("Start writing to", opp)
        #self._write_df_windows(processed_df, opp)
        #self._write_log(result_text_list, r_opp)
        # return processed_df

    def start_all(self, df_path_list):
        for df_path in df_path_list:
            self.start_one(df_path)
            print("===============")