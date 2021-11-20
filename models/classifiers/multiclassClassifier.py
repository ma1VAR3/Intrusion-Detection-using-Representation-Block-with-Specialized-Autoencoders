
from os import name


class MulticlassClassifier:
    def __init__(self, encoders, feature_dim, num_classes, epochs, batch_size):
        self.encoders = encoders
        self.feature_dim = feature_dim
        self.num_classes= num_classes
        self.epochs = epochs
        self.batch_size = batch_size
    
    def build_model(self):

        import tensorflow.keras as k
        from tensorflow.keras import layers
        from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
        from tensorflow.keras.models import Model
        from tensorflow.keras.utils import plot_model
        import tensorflow_addons as tfa
        
        input_layer = Input(shape=(self.feature_dim, ))
        rep_layers=[]
        dist_ops = []
        for i in range(len(self.encoders)):
            encoding = self.encoders[i](input_layer, training=False)
            
            feat_layer1 = Dense(self.feature_dim , activation="swish", name="feature_extractor"+str(i))(encoding)
            feat_layer2 = Dense(self.feature_dim , activation="swish", name="distribution_learner"+str(i))(encoding)
            dist_opt = Dense(1, activation="sigmoid", name="category_identifier"+str(i))(feat_layer2)
            rep_layer = layers.concatenate([dist_opt, feat_layer1])
            rep_layers.append(rep_layer)
            dist_ops.append(dist_opt)

        concat_layer = layers.concatenate([l for l in rep_layers], name="concatenation")
        dist_concat = layers.concatenate([l for l in dist_ops])
        layer1 = Dense(32, activation="swish", name="fully_connected")(concat_layer)
        layer1 = BatchNormalization()(layer1)

        output_layer = Dense(self.num_classes, activation="softmax")(layer1)

        classifier = Model(inputs=input_layer ,outputs=output_layer)
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
        classifier.summary()
        file = 'model.png'
        plot_model(classifier, to_file=file, show_shapes=True)
        self.classifier = classifier
        self.temp = Model(inputs=input_layer, outputs=dist_concat)
        
        
    def train(self, x_train, y_train, x_test, y_test):
        import tensorflow as tf

        self.build_model()

        def LRschedulerAE(epoch):
            import math
            initial_lrate = 0.01
            drop = 0.005
            epochs_drop = 5.0
            lrate = initial_lrate * math.pow(drop,  
                math.floor((1+epoch)/epochs_drop))
            return lrate

        clf_lr = tf.keras.callbacks.LearningRateScheduler(LRschedulerAE)

        history = self.classifier.fit(x_train, y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    callbacks=[clf_lr],
                    verbose=1).history

        from sklearn.metrics import f1_score
        import numpy as np

        y_preds = self.classifier.predict(x_test)
        y_preds = np.round_(y_preds)
        print(f1_score(y_test, y_preds, average='micro', zero_division=0))
        print(f1_score(y_test, y_preds, average='weighted', zero_division=0))

        y_p_t =  self.temp.predict(x_train[0:32])
        for i in range (32):
            print("predict:", y_p_t[i])
            print("label:", y_train[i])
        print("*"*20)
        y_p_t =  self.temp.predict(x_train[10000:10032])
        for i in range (32):
            print("predict:", y_p_t[i])
            print("label:", y_train[10000+i])
        print("*"*20)
        y_p_t =  self.temp.predict(x_train[20000:20032])
        for i in range (32):
            print("predict:", y_p_t[i])
            print("label:", y_train[20000+i])

        return history
    
    def load():
        pass

    def predict(self, x_vals):
        y_preds = self.classifier.predict(x_vals)

        return y_preds

    def evaluate(y_true, y_preds):
        from sklearn.metrics import f1_score
        import numpy as np
        
        y_preds = np.round_(y_preds)

        print(f1_score("Micro average F1 Score: ", y_true, y_preds, average='micro', zero_division=0))
        print(f1_score("Weighted average F1 Score: ", y_true, y_preds, average='weighted', zero_division=0))

    