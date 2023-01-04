from tensorflow import keras

class NeuralMF:
    def __init__(self, 
            num_users, num_items, regs=0, # Global Setting Value
            latent_features=8, # GMF Setting Value
            layers=[64,32,16,8], # MLP Setting Value
            ):
        num_layers = len(layers)

        # 1. 입력 층 생성
        user_input = keras.Input(shape=(1,), dtype='int32')
        item_input = keras.Input(shape=(1,), dtype='int32')

        # 2-1. GMF Layer Group
        user_embedding_gmf = keras.layers.Embedding(
                    num_users,
                    latent_features,
                    embeddings_regularizer=keras.regularizers.l2(regs),
                    name='gmf-user_emebedding')(user_input)
        item_embedding_gmf = keras.layers.Embedding(
                            num_items,
                            latent_features,
                            embeddings_regularizer=keras.regularizers.l2(regs),
                            name='gmf-item_emebedding')(item_input)
        user_latent_gmf = keras.layers.Flatten()(user_embedding_gmf)
        item_latent_gmf = keras.layers.Flatten()(item_embedding_gmf)
        result_gmf = keras.layers.Multiply()([user_latent_gmf, item_latent_gmf])

        # 2-2. MLP Layer Group
        user_embedding_mlp = keras.layers.Embedding(
                    num_users,
                    int(layers[0] / 2),
                    embeddings_regularizer=keras.regularizers.l2(regs),
                    name='mlp-user_emebedding')(user_input)
        item_embedding_mlp = keras.layers.Embedding(
                            num_items,
                            int(layers[0] / 2),
                            embeddings_regularizer=keras.regularizers.l2(regs),
                            name='mlp-item_emebedding')(item_input)
        user_latent_mlp = keras.layers.Flatten()(user_embedding_mlp)
        item_latent_mlp = keras.layers.Flatten()(item_embedding_mlp)
        result_mlp = keras.layers.Concatenate()([user_latent_mlp, item_latent_mlp])

        # ~ 2-2. MLP Hidden Layer
        for index in range(num_layers):
            hidden_layer = keras.layers.Dense(
                            layers[index],
                            kernel_regularizer=keras.regularizers.l2(regs),
                            activation = keras.activations.relu,
                            name = "mlp-layer_{}".format(index)
                        )
            result_mlp = hidden_layer(result_mlp)
        
        # 3. concat GMF output and MLP output
        concat = keras.layers.Concatenate()([result_gmf, result_mlp])

        # 4. Dense Layer
        output = keras.layers.Dense(1,kernel_initializer=keras.initializers.lecun_uniform(),name="output")(concat)

        self.model = keras.Model(inputs=[user_input, item_input], outputs=[output])
    
    def get_model(self):
        model = self.model
        return model




                


        