"""
Builder consists of the helper functions used to build the data set for 
analysis. 

Functions
    explode
    build_chunked
    build_interactions
    build_users
    build_nodes
    full_network
    user_info
"""
import pandas as pd
import numpy as np
import ast
import os
from itertools import groupby

lang_map = {
    "fa": "persian",
    "ar": "arabic",
    "und": "undefined",
    "cs": "czech",
    "cy": "welsh",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "fi": "finnish",
    "fr": "french",
    "in": "indonesian",
    "it": "italian",
    "nl": "dutch",
    "no": "norwegian",
    "ru": "russian",
    "tl": "tagalog",
    "tr": "turkish",
    "pt": "portugese",
    "ur": "urdu",
    "et": "estonian",
    "ro": "romanian",
    "pl": "polish",
    "sl": "slovenian",
    "lt": "lithuanian",
    "ca": "catalan",
    "is": "icelandic",
    "sd": "sindhi",
    "iw": "hebrew",
    "eu": "basque",
    "hi": "hindi",
    "ht": "haitian",
    "hu": "hungarian",
    "lv": "latvian",
    "sv": "swedish",
    "vi": "vietnamese",
    "ja": "japanese",
    "sk": "slovak",
    "zh": "chinese",
    "uk": "ukranian",
    "bs": "bosnian",
    "th": "thai",
    "sr": "serbian",
    "ps": "pashto",
    "id": "indonesian",
    "ko": "korean",
    "hr": "croatian",
    "si": "sinhala",
    "bn": "bengali",
    "ne": "nepali",
    "bg": "bulgarian",
    "mr": "marathi",
    "ckb": "central kurdish",
    "ta": "tamil",
    "te": "telugu",
    "ug": "uighur",
    "gu": "gujarati",
    "pa": "punjabi",
    "ml": "malayalam",
    "hy": "armenian",
    "ka": "georgian",
    "kn": "kannada",
    "my": "burmese",
    "am": "amharic",
    "he": "hebrew",
    "chr": "cherokee",
    "bo": "tibetan",
    "km": "central khmer",
    "iu": "inuktitut",
}


def build_chunked(directory):
    files = os.listdir(directory)
    retweets = pd.DataFrame()
    replies = pd.DataFrame()
    mentions = pd.DataFrame()
    tweets = pd.DataFrame()
    hashtags = pd.DataFrame()
    for f in files:
        chunked = pd.read_csv(
            directory + f, low_memory=False, chunksize=500000
        )
        for df in chunked:
            retweets = retweets.append(
                df.groupby(["userid", "retweet_userid"])["tweetid"]
                .count()
                .reset_index()
            )
            replies = replies.append(
                df.groupby(["userid", "in_reply_to_userid"])["tweetid"]
                .count()
                .reset_index()
            )
            mentions = mentions.append(
                df.groupby(["userid", "user_mentions"])["tweetid"]
                .count()
                .reset_index()
            )
            tweets = tweets.append(
                df[["userid", "tweetid", "tweet_language"]]
                .drop_duplicates()
                .groupby(["userid", "tweet_language"])["tweetid"]
                .count()
                .rename("tweets")
                .reset_index()
            )
            temp = df[["userid", "tweetid", "hashtags"]].drop_duplicates()
            temp.loc[temp["hashtags"].str.len() == 2, "hashtags"] = np.nan
            temp = temp.dropna(subset=["hashtags"])
            temp["hashtags"] = temp["hashtags"].apply(
                lambda x: x.strip("[").strip("]").split(",")
            )
            temp = temp.explode("hashtags").reset_index(drop=True)
            temp["hashtags"] = (
                temp["hashtags"].str.strip("'").str.strip().str.strip("'")
            )
            hashtags = hashtags.append(
                temp.groupby(["userid", "hashtags"])["tweetid"]
                .count()
                .rename("tweets")
                .reset_index()
            )
            del df
        del chunked
    retweets = (
        retweets.groupby(["userid", "retweet_userid"])["tweetid"]
        .sum()
        .reset_index()
    )
    replies = (
        replies.groupby(["userid", "in_reply_to_userid"])["tweetid"]
        .sum()
        .reset_index()
    )
    mentions = (
        mentions.groupby(["userid", "user_mentions"])["tweetid"]
        .sum()
        .reset_index()
    )
    tweets = (
        tweets.groupby(["userid", "tweet_language"])["tweets"]
        .sum()
        .reset_index()
    )
    hashtags = (
        hashtags.groupby(["userid", "hashtags"])["tweets"].sum().reset_index()
    )
    return retweets, replies, mentions, tweets, hashtags


def build_interactions(directories):
    retweets, replies, mentions, tweets, hashtags = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    for d in directories:
        r, rp, m, t, h = build_chunked(d)
        retweets, replies, mentions, tweets, hashtags = (
            retweets.append(r),
            replies.append(rp),
            mentions.append(m),
            tweets.append(t),
            hashtags.append(h),
        )
    mentions = mentions[mentions["user_mentions"] != "[]"]
    mentions["user_mentions"] = mentions["user_mentions"].apply(
        lambda x: x.strip("[").strip("]").split(",")
    )
    mentions = mentions.explode("user_mentions")
    mentions = (
        mentions.groupby(["userid", "user_mentions"])["tweetid"]
        .sum()
        .reset_index()
    )
    retweets.columns = ["source", "target", "retweets"]
    replies.columns = ["source", "target", "replies"]
    mentions.columns = ["source", "target", "mentions"]
    interactions = retweets.merge(
        replies, on=["source", "target"], how="outer"
    )
    interactions = interactions.merge(
        mentions, on=["source", "target"], how="outer"
    )
    interactions = interactions.fillna(0)
    interactions["total"] = interactions[
        ["retweets", "replies", "mentions"]
    ].sum(axis=1)
    interactions["source"] = interactions["source"].str.strip("'")
    interactions["target"] = interactions["target"].str.strip("'")
    interactions["source"] = interactions["source"].astype(str)
    interactions["target"] = interactions["target"].astype(str)
    tweets["tweet_language"] = tweets["tweet_language"].map(lang_map)
    return interactions, hashtags, tweets


def build_users(files):
    users = pd.DataFrame()
    for f in files:
        users = users.append(pd.read_csv(f))
    return users


def build_nodes(users, interactions=None):
    nodes = []
    node_users = []
    for u in users["userid"].unique():
        node_users.append(u)
        nodes.append((u, {"account": "removed"}))
    if type(interactions) != pd.core.frame.DataFrame:
        return nodes
    source = list(interactions["source"].unique())
    source = [u for u in source if u not in node_users]
    for u in source:
        nodes.append((u, {"account": "tweet"}))
    target = list(interactions["target"].unique())
    target = [u for u in target if u not in node_users and u not in source]
    for u in target:
        nodes.append((u, {"account": "interacted"}))
    return nodes


def full_network():
    users_files = ["../data/iran/iran_users.csv"]
    iran_users = build_users(users_files)
    directories = ["../data/iran/tweets/"]
    ir_interactions, ir_hashtags, ir_tweets = build_interactions(directories)
    ir_interactions["country"] = "iran"
    ir_hashtags["country"] = "iran"
    ir_tweets["country"] = "iran"
    iran_users = build_nodes(iran_users, ir_interactions)
    users_files = [
        "../data/russia/russia_users.csv",
        "../data/ira/ira_users.csv",
    ]
    russian_users = build_users(users_files)
    directories = ["../data/russia/tweets/", "../data/ira/tweets/"]
    ru_interactions, ru_hashtags, ru_tweets = build_interactions(directories)
    ru_interactions["country"] = "russia"
    ru_hashtags["country"] = "russia"
    ru_tweets["country"] = "russia"
    russian_users = build_nodes(russian_users, ru_interactions)
    interactions = ir_interactions.append(ru_interactions)
    hashtags = ir_hashtags.append(ru_hashtags)
    tweets = ir_tweets.append(ru_tweets)
    for u in iran_users:
        u[1]["country"] = "iran"
    for u in russian_users:
        u[1]["country"] = "russia"
    users = iran_users
    users.extend(russian_users)
    return users, interactions, hashtags, primary_lang(tweets)


def user_info():
    users_files = ["../data/iran/iran_users.csv"]
    iran_users = build_users(users_files)
    users_files = [
        "../data/russia/russia_users.csv",
        "../data/ira/ira_users.csv",
    ]
    russian_users = build_users(users_files)
    iran_users["country"] = "iran"
    russian_users["country"] = "russia"
    users = iran_users.append(russian_users)
    users = users.drop(
        [
            "user_display_name",
            "user_screen_name",
            "user_reported_location",
            "user_profile_description",
            "user_profile_url",
            "file",
        ],
        axis=1,
    )
    return users


def primary_lang(tweets):
    main_languages = ["english", "persian", "russian", "arabic"]
    for l in main_languages:
        tweets.loc[tweets["tweet_language"] == l, "lang"] = l
    tweets["lang"] = tweets["lang"].fillna("other")
    tweets = (
        tweets.groupby(["userid", "country", "lang"])["tweets"]
        .sum()
        .reset_index()
    )
    primary_lang = tweets[
        tweets.groupby(["userid"])["tweets"].transform(max) == tweets["tweets"]
    ]
    primary_users = primary_lang.groupby(["userid"])["lang"].count()
    nonmulti = primary_lang[
        primary_lang["userid"].isin(primary_users[primary_users == 1].index)
    ]
    multi = primary_lang[
        primary_lang["userid"].isin(primary_users[primary_users > 1].index)
    ]
    multi = multi.groupby(["userid", "country"])["lang"].unique().reset_index()
    multi["lang"] = multi["lang"].apply(lang_mapper)
    primary_lang = nonmulti.drop(["tweets"], axis=1).append(multi)
    primary_lang.columns = ["userid", "country", "primary_lang"]
    tweets = tweets.merge(primary_lang, on=["userid", "country"], how="left")
    return tweets


def lang_mapper(x):
    if "english" in x:
        return "english"
    elif "russian" in x:
        return "russian"
    elif "persian" in x:
        return "persian"
    elif "arabic" in x:
        return "arabic"
    else:
        return "other"
