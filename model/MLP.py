from tensorflow import keras

class MLP:
    def __init__(self,num_users,num_items,layers=[64,32,16,8],regs=0):
        num_layers = len(layers)
        
        # 1. 입력 층 생성
        user_input = keras.Input(shape=(1,), dtype='int32')
        item_input = keras.Input(shape=(1,), dtype='int32')

        # 2. embedding 층 생성
        user_embedding = keras.layers.Embedding(
                            num_users,
                            int(layers[0] / 2),
                            embeddings_regularizer=keras.regularizers.l2(regs),
                            name='user_emebedding')(user_input)
        item_embedding = keras.layers.Embedding(
                            num_items,
                            int(layers[0] / 2),
                            embeddings_regularizer=keras.regularizers.l2(regs),
                            name='item_emebedding')(item_input)

        # 3. Flatten 층
        user_latent = keras.layers.Flatten()(user_embedding)
        item_latent = keras.layers.Flatten()(item_embedding)

        # 5. concat : layer 0, size : layer [0]
        vector = keras.layers.Concatenate()([user_latent, item_latent])

        # 6. Hidden Layers : 1 ~ num_layer
        for index in range(num_layers):
            hidden_layer = keras.layers.Dense(
                            layers[index],
                            kernel_regularizer=keras.regularizers.l2(regs),
                            activation = keras.activations.relu,
                            name = "layer_{}".format(index)
                        )
            vector = hidden_layer(vector)

        
        # 7. Dense Layer
        output = keras.layers.Dense(1,kernel_initializer=keras.initializers.lecun_uniform(),name="output")(vector)

        self.model = keras.Model(inputs=[user_input, item_input], outputs=[output])

    def get_model(self):
        model = self.model
        return model
