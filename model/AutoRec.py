from tensorflow import keras
    
class AutoRec(keras.Model):
    def __init__(self, num_features, reg=0.1, layers=[300], **kwargs):
        super().__init__(kwargs)

        self.num_features = num_features
        self.num_layers = len(layers)
        self.num_neuron = layers
        self.layer_list = []

        # Latent Layer
        for index in range(self.num_layers):
            layer = keras.layers.Dense(
                self.num_neuron[index],
                kernel_regularizer=keras.regularizers.l2(reg),
                activation='relu',
                name='layer{}'.format(index)
            )
            self.layer_list.append(layer)

        # 동일한 출력을 내보내는 목적을 가질 Output Layer
        output_layer = keras.layers.Dense(num_features, kernel_regularizer=keras.regularizers.l2(reg))
        self.layer_list.append(output_layer)

    def call(self, inputs):
        result = inputs
        for layer in self.layer_list:
            result = layer(result)
        return result
        