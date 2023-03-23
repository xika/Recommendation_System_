#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt


# In[2]:


dataset_path = Path('ml-1m')


# In[3]:


users = pd.read_csv(
    dataset_path/"users.dat",
    sep="::",
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
    encoding='latin-1',
    engine='python'
)

ratings = pd.read_csv(
    dataset_path/"ratings.dat",
    sep="::",
    names=["user_id", "movie_id", "rating", "unix_timestamp"],
    encoding='latin-1',
    engine='python'
)

movies = pd.read_csv(
    dataset_path/"movies.dat", sep="::", names=["movie_id", "title", "genres"],
    encoding='latin-1',
    engine='python'
)


# In[4]:


ratings_df = pd.merge(ratings, movies)[['user_id', 'title', 'rating', 'unix_timestamp']]


# In[5]:


ratings_df["user_id"] = ratings_df["user_id"].astype(str)


# In[6]:


ratings_per_user = ratings_df.groupby('user_id').rating.count()
ratings_per_item = ratings_df.groupby('title').rating.count()

print(f"Total No. of users: {len(ratings_df.user_id.unique())}")
print(f"Total No. of items: {len(ratings_df.title.unique())}")
print("\n")

print(f"Max observed rating: {ratings_df.rating.max()}")
print(f"Min observed rating: {ratings_df.rating.min()}")
print("\n")

print(f"Max no. of user ratings: {ratings_per_user.max()}")
print(f"Min no. of user ratings: {ratings_per_user.min()}")
print(f"Median no. of ratings per user: {ratings_per_user.median()}")
print("\n")

print(f"Max no. of item ratings: {ratings_per_item.max()}")
print(f"Min no. of item ratings: {ratings_per_item.min()}")
print(f"Median no. of ratings per item: {ratings_per_item.median()}")


# In[7]:


def get_last_n_ratings_by_user(
    df, n, min_ratings_per_user=1, user_colname="user_id", timestamp_colname="unix_timestamp"
):
    return (
        df.groupby(user_colname)
        .filter(lambda x: len(x) >= min_ratings_per_user)
        .sort_values(timestamp_colname)
        .groupby(user_colname)
        .tail(n)
        .sort_values(user_colname)
    )


# In[8]:


get_last_n_ratings_by_user(ratings_df, 1)


# In[9]:


def mark_last_n_ratings_as_validation_set(
    df, n, min_ratings=1, user_colname="user_id", timestamp_colname="unix_timestamp"
):
    """
    Mark the chronologically last n ratings as the validation set.
    This is done by adding the additional 'is_valid' column to the df.
    :param df: a DataFrame containing user item ratings
    :param n: the number of ratings to include in the validation set
    :param min_ratings: only include users with more than this many ratings
    :param user_id_colname: the name of the column containing user ids
    :param timestamp_colname: the name of the column containing the imestamps
    :return: the same df with the additional 'is_valid' column added
    """
    df["is_valid"] = False
    df.loc[
        get_last_n_ratings_by_user(
            df,
            n,
            min_ratings,
            user_colname=user_colname,
            timestamp_colname=timestamp_colname,
        ).index,
        "is_valid",
    ] = True

    return df


# In[10]:


mark_last_n_ratings_as_validation_set(ratings_df, 1)


# In[11]:


train_df = ratings_df[ratings_df.is_valid==False]
valid_df = ratings_df[ratings_df.is_valid==True]


# In[12]:


len(valid_df)


# In[47]:


from sklearn.preprocessing import LabelEncoder


# In[48]:


le = LabelEncoder()


# In[49]:


users['sex_encoded'] = le.fit_transform(users.sex)


# In[50]:


users['age_group_encoded'] = le.fit_transform(users.age_group)


# In[51]:


users["user_id"] = users["user_id"].astype(str)


# In[52]:


seq_with_user_features = pd.merge(seq_df, users)


# In[53]:


train_df = seq_with_user_features[seq_with_user_features.is_valid == False]
valid_df = seq_with_user_features[seq_with_user_features.is_valid == True]


# In[54]:


class MovieSequenceDataset(Dataset):
    def __init__(self, df, movie_lookup, user_lookup):
        super().__init__()
        self.df = df
        self.movie_lookup = movie_lookup
        self.user_lookup = user_lookup

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        user_id = self.user_lookup[str(data.user_id)]
        movie_ids = torch.tensor([self.movie_lookup[title] for title in data.title])

        previous_ratings = torch.tensor(
            [rating if rating != "[PAD]" else 0 for rating in data.previous_ratings]
        )

        attention_mask = torch.tensor(data.pad_mask)
        target_rating = data.target_rating
        encoded_features = {
            "user_id": user_id,
            "movie_ids": movie_ids,
            "ratings": previous_ratings,
            "age_group": data["age_group_encoded"],
            "sex": data["sex_encoded"],
            "occupation": data["occupation"],
        }

        return (encoded_features, attention_mask), torch.tensor(
            target_rating, dtype=torch.float32
        )


# In[55]:


train_dataset = MovieSequenceDataset(train_df, movie_lookup, user_lookup)
valid_dataset = MovieSequenceDataset(valid_df, movie_lookup, user_lookup)


# In[56]:


class BstTransformer(nn.Module):
    def __init__(
        self,
        movies_num_unique,
        users_num_unique,
        sequence_length=10,
        embedding_size=120,
        num_transformer_layers=1,
        ratings_range=(0.5, 5.5),
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.y_range = ratings_range
        self.movies_embeddings = nn.Embedding(
            movies_num_unique + 1, embedding_size, padding_idx=0
        )
        self.user_embeddings = nn.Embedding(users_num_unique + 1, embedding_size)
        self.ratings_embeddings = nn.Embedding(6, embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(sequence_length, embedding_size)

        self.sex_embeddings = nn.Embedding(
            3,
            2,
        )
        self.occupation_embeddings = nn.Embedding(
            22,
            11,
        )
        self.age_group_embeddings = nn.Embedding(
            8,
            4,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=12,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=num_transformer_layers,
        )

        self.linear = nn.Sequential(
            nn.Linear(
                embedding_size + (embedding_size * sequence_length) + 4 + 11 + 2,
                1024,
            ),
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        features, mask = inputs

        user_id = self.user_embeddings(features["user_id"])

        age_group = self.age_group_embeddings(features["age_group"])
        sex = self.sex_embeddings(features["sex"])
        occupation = self.occupation_embeddings(features["occupation"])

        user_features = user_features = torch.cat(
            (user_id, sex, age_group, occupation), 1
        )

        movie_history = features["movie_ids"][:, :-1]
        target_movie = features["movie_ids"][:, -1]

        ratings = self.ratings_embeddings(features["ratings"])

        encoded_movies = self.movies_embeddings(movie_history)
        encoded_target_movie = self.movies_embeddings(target_movie)

        positions = torch.arange(
            0,
            self.sequence_length - 1,
            1,
            dtype=int,
            device=features["movie_ids"].device,
        )
        positions = self.position_embeddings(positions)

        encoded_sequence_movies_with_position_and_rating = (
            encoded_movies + ratings + positions
        )
        encoded_target_movie = encoded_target_movie.unsqueeze(1)

        transformer_features = torch.cat(
            (encoded_sequence_movies_with_position_and_rating, encoded_target_movie),
            dim=1,
        )
        transformer_output = self.encoder(
            transformer_features, src_key_padding_mask=mask
        )
        transformer_output = torch.flatten(transformer_output, start_dim=1)

        combined_output = torch.cat((transformer_output, user_features), dim=1)

        rating = self.linear(combined_output)
        rating = rating.squeeze()
        if self.y_range is None:
            return rating
        else:
            return rating * (self.y_range[1] - self.y_range[0]) + self.y_range[0]


# In[57]:


notebook_launcher(train_seq_model, num_processes=0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




