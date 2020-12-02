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

# https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows/40449726#40449726
def explode(df, lst_cols, fill_value='', preserve_index=False):
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    idx_cols = df.columns.difference(lst_cols)
    lens = df[lst_cols[0]].str.len()
    idx = np.repeat(df.index.values, lens)
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    if (lens == 0).any():
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    res = res.sort_index()
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res

def build_chunked(directory):
    files = os.listdir(directory)
    retweets = pd.DataFrame()
    replies = pd.DataFrame()
    mentions = pd.DataFrame()
    for f in files:
        chunked = pd.read_csv(directory + f, low_memory=False, chunksize=500000)
        for df in chunked:
            retweets = retweets.append(df.groupby(['userid','retweet_userid'])['tweetid'].count().reset_index())
            replies = replies.append(df.groupby(['userid','in_reply_to_userid'])['tweetid'].count().reset_index())
            mentions = mentions.append(df.groupby(['userid','user_mentions'])['tweetid'].count().reset_index())
            del df
        del chunked
    retweets = retweets.groupby(['userid','retweet_userid'])['tweetid'].sum().reset_index()
    replies = replies.groupby(['userid','in_reply_to_userid'])['tweetid'].sum().reset_index()
    mentions = mentions.groupby(['userid','user_mentions'])['tweetid'].count().reset_index()
    return retweets, replies, mentions

def build_interactions(directories):
    retweets, replies, mentions = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for d in directories:
        r, rp, m = build_chunked(d)
        retweets, replies, mentions = retweets.append(r), replies.append(rp), mentions.append(m)
    mentions = mentions[mentions['user_mentions'] != '[]']
    mentions['user_mentions'] = mentions['user_mentions'].apply(lambda x: x.strip('[').strip(']').split(','))
    mentions = explode(mentions, ['user_mentions'], fill_value=np.nan)
    mentions = mentions.groupby(['userid','user_mentions'])['tweetid'].sum().reset_index()
    retweets.columns = ['source','target','retweets']
    replies.columns = ['source','target','replies']
    mentions.columns = ['source','target','mentions']
    interactions = retweets.merge(replies, on=['source','target'], how='outer')
    interactions = interactions.merge(mentions, on=['source','target'], how='outer')
    interactions = interactions.fillna(0)
    interactions['total'] = interactions[['retweets','replies','mentions']].sum(axis=1)
    interactions['source'] = interactions['source'].str.strip("'")
    interactions['target'] = interactions['target'].str.strip("'")
    interactions['source'] = interactions['source'].astype(str)
    interactions['target'] = interactions['target'].astype(str)
    return interactions

def build_users(files):
    users = pd.DataFrame()
    for f in files:
        users = users.append(pd.read_csv(f))
    return users

def build_nodes(users, interactions=None):
    nodes = []
    node_users = []
    for u in users['userid'].unique():
        node_users.append(u)
        nodes.append((u, {'account':'removed'}))
    if type(interactions) != pd.core.frame.DataFrame:
        return nodes
    source = list(interactions['source'].unique())
    source = [u for u in source if u not in node_users]
    for u in source:
        nodes.append((u, {'account':'tweet'}))
    target = list(interactions['target'].unique())
    target = [u for u in target if u not in node_users and u not in source]
    for u in target:
        nodes.append((u, {'account':'interacted'}))
    return nodes

def full_network():
    users_files = ['data/iran/iran_users.csv']
    iran_users = build_users(users_files)
    directories = ['data/iran/tweets/']
    ir_interactions = build_interactions(directories)
    ir_interactions['country'] = 'iran'
    iran_users = build_nodes(iran_users, ir_interactions)
    users_files = ['data/russia/russia_users.csv','data/ira/ira_users.csv']
    russian_users = build_users(users_files)
    directories = ['data/russia/tweets/','data/ira/tweets/']
    ru_interactions = build_interactions(directories)
    ru_interactions['country'] = 'russia'
    russian_users = build_nodes(russian_users, ru_interactions)
    interactions = ir_interactions.append(ru_interactions)
    for u in iran_users:
        u[1]['country'] = 'iran'
    for u in russian_users:
        u[1]['country'] = 'russia'
    users = iran_users
    users.extend(russian_users)
    return users, interactions

def user_info():
    users_files = ['data/iran/iran_users.csv']
    iran_users = build_users(users_files)
    users_files = ['data/russia/russia_users.csv','data/ira/ira_users.csv']
    russian_users = build_users(users_files)
    iran_users['country'] = 'iran'
    russian_users['country'] = 'russia'
    users = iran_users.append(russian_users)
    users = users.drop(['user_display_name', 'user_screen_name',
    'user_reported_location', 'user_profile_description','user_profile_url',
    'file'], axis=1)
    return users