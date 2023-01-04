import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

PATH = "datas/movielens_small/{}"

class DataLoader():
    def __init__(self):
        ratings_df = pd.read_csv(PATH.format("ratings.csv"), encoding='utf-8')

        # split train(80%) test(20%)
        self.train_df, self.test_df = train_test_split(ratings_df, test_size=0.2, random_state=42, shuffle=True)

        # user dataset setting
        self.users = self.train_df["userId"].unique()
        self.num_users = len(self.users)
        self.user_to_index = { user : idx for idx, user in enumerate(self.users) }

        # movie dataset setting
        self.movies = self.train_df['movieId'].unique()
        self.num_items = len(self.movies)
        self.movie_to_index = { movie : idx for idx, movie in enumerate(self.movies) }

        # 테스트 데이터 셋은 오로지 훈련데이터 속에 존재하는 사용자, 그리고 영화 데이터에 존재해야 한다.
        self.test_df = self.test_df[self.test_df['userId'].isin(self.users) & self.test_df['movieId'].isin(self.movies)]

    def generate_trainset(self):
        # 인덱스 타입으로 변환
        X_train = pd.DataFrame({
                        'user': self.train_df['userId'].map(self.user_to_index),
                        'movie': self.train_df['movieId'].map(self.movie_to_index)
                    })
        y_train = self.train_df['rating'].astype(np.float32)
        
        return X_train.to_numpy(), y_train.to_numpy()

    def generate_testset(self):
        # 인덱스 타입으로 변환
        X_test = pd.DataFrame({
                        'user': self.test_df['userId'].map(self.user_to_index),
                        'movie': self.test_df['movieId'].map(self.movie_to_index)
                    })
        y_test = self.test_df['rating'].astype(np.float32)
        
        return X_test.to_numpy(), y_test.to_numpy()