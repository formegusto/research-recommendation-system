import tensorflow as tf
from tensorflow import keras

def print_status_bar(iteration, total, loss, metrics = None):
    metrics = " - ".join([f"{m.name}: {m.result():.4f}"
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print(f"\r{iteration}/{total}  " + metrics ,
          end = end)

class FactorizationMachine(keras.Model):
    # train_set = (X_train, y_train)
    # test_set = (X_test, y_test)
    def __init__(self, train_set, test_set, 
                batch_size=32,
                learner="adam",
                learning_rate=0.01,
                num_factor=8, **kwargs):
        super().__init__(**kwargs)

        X_train, y_train = train_set
        X_test, y_test = test_set

        # 1. tensor variable ref.Factorization Machine Model Operation
        self.set_variables(X_train, num_factor)

        # 2. set tensor slices data, use batch data
        self.set_dataset(X_train, y_train, X_test, y_test, batch_size)

        # 3. activation, loss function, and optimizer
        self.set_etc(learner, learning_rate)

        self.batch_size = batch_size
        self.num_train = y_train.size
        self.num_test = y_test.size

    # 1. tensor variable ref.Factorization Machine Model Operation
    def set_variables(self, X_train, num_factor):
        _, p = X_train.shape

        # Global Bias
        self.w_0 = tf.Variable([0.0], name="Global Bias")

        # i 번째 개별 **특성**에 대한 가중치
        self.w = tf.Variable(tf.zeros(shape=[p]), name="Weights")

        # v_i, v_factor 는 f개의 latent factor로 표현된 2-way interaction을 계산하는 내적을 의미
        self.v = tf.Variable(tf.random.normal(shape=(p, num_factor)), name="Latent Factor")

    # 2. set tensor slices data, use batch data
    def set_dataset(self, X_train, y_train, X_test, y_test, batch_size):   
        self.train_data = tf.data.Dataset.from_tensor_slices(
            (tf.cast(X_train, tf.float32), tf.cast(y_train, tf.float32))
        ).shuffle(500).batch(batch_size)
        self.test_data = tf.data.Dataset.from_tensor_slices(
            (tf.cast(X_test, tf.float32), tf.cast(y_test, tf.float32))
        ).shuffle(200).batch(batch_size)

    # 3. activation, loss function, and optimizer
    def set_etc(self, learner, learning_rate):
        if learner == "adagrad":
            self.optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif learner == "rmsprop":
            self.optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif learner == "adam":
            self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

        self.loss_fn_ = keras.losses.binary_crossentropy
        self.mean_loss_ = keras.metrics.Mean()
        self.metrics_ = [keras.metrics.BinaryAccuracy()]
        self.test_acc_ = keras.metrics.BinaryAccuracy()

    # *. predict function, keras.Model.call function ref.Factorization Machine Model Operation
    # = tensorflow model.call is "value input to model", X_batch data in
    def call(self, inputs):
        # 시그마(W_i, x_i)
        degree_1 = tf.reduce_sum(tf.multiply(self.w, inputs), axis=1)

        # 오른쪽에 어려운 식 계산
        degree_2 = 0.5 * tf.reduce_sum(
                tf.math.pow(tf.matmul(inputs, self.v), 2)
                - tf.matmul(tf.math.pow(inputs, 2), tf.math.pow(self.v, 2)),
                1, keepdims=False
            )

        # 나머지 계산 (Sigmoid)
        predict = tf.math.sigmoid(self.w_0 + degree_1, degree_2)

        return predict

    def fit(self, epochs=10):
        n_steps = self.num_train // self.batch_size

        for epoch in range(epochs):
            print("epoch : {} / {}".format(epoch + 1, epochs))
            predicts = list()
            for step, (X_batch, y_batch) in enumerate(self.train_data):
                with tf.GradientTape() as tape:
                    predict = self(X_batch)
                    loss_value = self.loss_fn_(y_batch, predict)
                gradients = tape.gradient(
                    loss_value, self.trainable_variables,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO
                )
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                self.mean_loss_(loss_value)

                for metric in self.metrics_:
                    metric(y_batch, predict)

                print_status_bar(step * self.batch_size, self.num_train, self.mean_loss_, metrics=self.metrics_)
            
            for x_test, y_test in self.test_data:
                prediction = self(x_test)
                predicts.append(prediction)
                self.test_acc_.update_state(y_test, prediction)

            print_status_bar(n_steps * self.batch_size, n_steps * self.batch_size, self.mean_loss_, metrics=self.metrics_)
            print("검증 정확도: {}".format(self.test_acc_.result().numpy()))

            for metric in [self.mean_loss_] + [self.test_acc_] + self.metrics_:
                metric.reset_states()
        
        self.predicts = predicts

        




        
