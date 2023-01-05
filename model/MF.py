import numpy as np
from sklearn.preprocessing import MinMaxScaler

# SGD 를 적용한 Matrix Factorization
class MatrixFactorization:
    def __init__(self, train_df, test_df, latent_factor=20, learning_rate=0.01, reg=0.01):
        # 1. Setting Value
        self.train_df = train_df.pivot_table("rating", index="userId", columns="movieId").fillna(0)
        self.test_df = test_df.pivot_table("rating", index="userId", columns="movieId").fillna(0)

        self.train_datas = self.train_df.to_numpy()
        self.test_datas = self.test_df.to_numpy()

        self.num_user, self.num_item = self.train_datas.shape
        self.learning_rate = learning_rate
        self.reg = reg

        self.set_latent_factor(latent_factor)
        self.set_confidence(train_df)
        self.set_bias()

    def set_latent_factor(self,latent_factor):
        # 2. Setting Latent Factor Matrix p_u(사용자), q_i(아이템)
        self.p_u = np.random.normal(size=(self.num_user, latent_factor))
        self.q_i = np.random.normal(size=(self.num_item, latent_factor))

    def set_confidence(self, param_train_df):
        # 3. Confidence ref.Inputs with varying confidence levels
        user_confidence = (self.train_df  > 0).sum(axis=1)
        item_confidence = (self.train_df > 0).sum(axis=0)

        confidences = param_train_df.apply(lambda x: 
                   user_confidence.loc[x['userId']] + 
                   item_confidence.loc[x['movieId']], axis=1)
        param_train_df['confidence'] = confidences
        confidence_df = param_train_df.pivot_table(values="confidence", index="userId", columns="movieId")
        confidence_df.fillna(1, inplace=True)
        confidence_matrix = confidence_df.to_numpy()

        # 4. Scalling
        scaler = MinMaxScaler()
        scaled_confidence = scaler.fit_transform(confidence_matrix)
        scaled_confidence += 0.0001
        self.confidence = scaled_confidence

    def set_bias(self):
        # 5. bias setting
        self.b_u = np.zeros(self.num_user)
        self.b_i = np.zeros(self.num_item)
        self.b = np.mean(self.train_datas[self.train_datas != 0])

    def fit(self,epoch=30):
        self.history = []
        for _epoch in range(epoch):
            for i_u in range(self.num_user):
                for i_i in range(self.num_item):
                    if self.train_datas[i_u, i_i] > 0:
                        self.gradient_descent(i_u, i_i, self.train_datas[i_u, i_i])
            _loss = self.loss()
            _test_loss = self.test_loss()
            self.history.append((_epoch, _loss, _test_loss))
            
            print("Epoch: {} ; loss = {} ; test_loss = {}".format(_epoch + 1, _loss, _test_loss))

    def get_each_prediction(self, i_u, i_i):
        return self.b + self.b_u[i_u] + self.b_i[i_i] + np.dot(self.p_u[i_u, :], self.q_i[i_i, :].T)

    def get_whole_prediction(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:,] + np.dot(self.p_u, self.q_i.T)

    
    def gradient_descent(self,i_u, i_i, rating):
        prediction = self.get_each_prediction(i_u, i_i)
        
        dbu = -2 * self.confidence[i_u][i_i] * (rating - self.b_u[i_u] - self.b_i[i_i] - self.b - prediction) + 2 * self.reg * self.b_u[i_u]
        dbi = -2 * self.confidence[i_u][i_i] * (rating - self.b_u[i_u] - self.b_i[i_i] - self.b - prediction) + 2 * self.reg * self.b_i[i_i]
        
        self.b_u[i_u] -= self.learning_rate * dbu
        self.b_i[i_i] -= self.learning_rate * dbi
        
        dp = -2 * self.confidence[i_u][i_i] * \
                (rating - self.b_u[i_u] - self.b_i[i_i] - self.b - prediction) * self.q_i[i_i,:] + 2 * (self.reg * self.p_u[i_u, :])
        dq = -2 * self.confidence[i_u][i_i] * \
                (rating - self.b_u[i_u] - self.b_i[i_i] - self.b - prediction) * self.p_u[i_u,:] + 2 * (self.reg * self.q_i[i_i, :])
        
        self.p_u[i_u, :] -= self.learning_rate * dp
        self.q_i[i_i, :] -= self.learning_rate * dq

    def test_loss(self):
        xi, yi = self.test_datas.nonzero()
        predicted = self.get_whole_prediction()
        test_loss = 0
        for x, y in zip(xi, yi):
            test_loss += np.power(self.test_datas[x, y] - predicted[x, y], 2)
        return np.sqrt(test_loss) / len(xi)

    def loss(self):
        xi, yi = self.train_datas.nonzero()
        predicted = self.get_whole_prediction()
        test_loss = 0
        for x, y in zip(xi, yi):
            test_loss += np.power(self.train_datas[x, y] - predicted[x, y], 2)
        return np.sqrt(test_loss) / len(xi)