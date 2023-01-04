from tensorflow import keras

class GMF:
    def __init__(self, num_users, num_items, latent_features=8, regs=0):
        # 1. 입력 층 생성
        user_input = keras.Input(shape=(1,), dtype='int32')
        item_input = keras.Input(shape=(1,), dtype='int32')

        # 2. embedding 층 생성
        user_embedding = keras.layers.Embedding(
                            num_users,
                            latent_features,
                            embeddings_regularizer=keras.regularizers.l2(regs),
                            name='user_emebedding')(user_input)
        item_embedding = keras.layers.Embedding(
                            num_items,
                            latent_features,
                            embeddings_regularizer=keras.regularizers.l2(regs),
                            name='item_emebedding')(item_input)

        # 3. Flatten 층
        user_latent = keras.layers.Flatten()(user_embedding)
        item_latent = keras.layers.Flatten()(item_embedding)

        # 4. concat with multiply
        concat = keras.layers.Multiply()([user_latent, item_latent])

        # 5. Dense Layer
        output = keras.layers.Dense(1, kernel_initializer=keras.initializers.lecun_uniform(), name='output')(concat)

        self.model = keras.Model(inputs=[user_input, item_input], outputs=[output])

    def get_model(self):
        model = self.model
        return model
