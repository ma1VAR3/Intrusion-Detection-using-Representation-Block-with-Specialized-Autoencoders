DEBUG = False
NUM_FOLDS = 3
feature_dim = 32
encoding_dim = 16
ae_epoch = 50
clf_epoch = 30
batch_size = 32
RANDOM_STATE = 2021
dataset = "nslkdd"

if DEBUG==True:
    NUM_FOLDS = 3
    ae_epoch = 1 
    clf_epoch = 1

def train_binary():
    from models.autoencoders.binaryAE import BinaryAutoencoder
    from models.classifiers.binaryClassifier import BinaryClassifier  
    import numpy as np
    from sklearn.model_selection import train_test_split
    from utils import getbinarydata

    encoders = []

    x_b, y_b, x_a, y_a, x, y, feats = getbinarydata(feature_dim, dataset)

    X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=.2, random_state=42)
    
    binary_ae_b = BinaryAutoencoder(inp_dim= feature_dim, enc_dim= encoding_dim, epochs= ae_epoch, batch_size=32)
    binary_ae_b.train(x_b)
    binary_ae_b.freeze_encoder()
    binary_encoder_b = binary_ae_b.encoder
    encoders.append(binary_encoder_b)

    binary_ae_a = BinaryAutoencoder(inp_dim= feature_dim, enc_dim= encoding_dim, epochs= ae_epoch, batch_size=32)
    binary_ae_a.train(x_a)
    binary_ae_a.freeze_encoder()
    binary_encoder_a = binary_ae_a.encoder
    encoders.append(binary_encoder_a)

    b_classifier = BinaryClassifier(encoders= encoders,feature_dim= feature_dim, epochs= clf_epoch, batch_size=32)
    history = b_classifier.train(X_train, y_train, X_valid, y_valid)

    b_classifier.classifier.save('./saved/b_classifier.h5')

    # enc_trainableParams = np.sum([np.prod(v.get_shape()) for v in binary_encoder.trainable_weights])
    # enc_nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in binary_encoder.non_trainable_weights])
    # enc_totalParams = enc_trainableParams + enc_nonTrainableParams
    
    clf_trainableParams = np.sum([np.prod(v.get_shape()) for v in b_classifier.classifier.trainable_weights])
    clf_nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in b_classifier.classifier.non_trainable_weights])
    clf_totalParams = clf_trainableParams + clf_nonTrainableParams
    
    # totalParams = enc_totalParams + clf_totalParams
    
    return history, clf_totalParams

def train_multi():
    from models.autoencoders.multiAE import MultiAutoencoder
    from models.classifiers.multiclassClassifier import MulticlassClassifier
    from keras.utils import np_utils
    from sklearn.model_selection import StratifiedKFold, train_test_split
    import numpy as np
    from utils import getcategorydata, getattackdata

    x_data_train, x_data_valid, y_data_train, y_data_valid, feats = getattackdata(feature_dim, "nslkdd")
    x, y = getcategorydata(feature_dim, "nslkdd", feats)

    encoders = []
    for i in range(len(x)):
        x_i = x[i]
        multi_ae = MultiAutoencoder(inp_dim= feature_dim, enc_dim= encoding_dim, epochs= ae_epoch, batch_size=32)
        multi_ae.train(x_i)
        multi_ae.freeze_encoder()
        multi_encoder = multi_ae.encoder
        encoders.append(multi_encoder)

    multi_classifier = MulticlassClassifier(encoders= encoders,feature_dim= feature_dim, num_classes = y_data_train.shape[1] ,epochs= clf_epoch, batch_size=32)
    history = multi_classifier.train(x_data_train, y_data_train, x_data_valid, y_data_valid)
    
    multi_classifier.classifier.save('./saved/multi_classifier.h5')

    clf_trainableParams = np.sum([np.prod(v.get_shape()) for v in multi_classifier.classifier.trainable_weights])
    clf_nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in multi_classifier.classifier.non_trainable_weights])
    clf_totalParams = clf_trainableParams + clf_nonTrainableParams

    

    return history, clf_totalParams


if __name__ == "__main__":
    
    
    
    # X_train, X_test, y_train, y_test = getdata()
    
    # 
    
    # print(X_bin.shape)
    # print(y_bin.shape)

    bin_history, bin_params = train_binary()

    
    
    multi_history, multi_params = train_multi()

    print("Binary classifier params: ", bin_params)
    print("Multiclass classifier params: ", multi_params)

    #  multi_history, multi_params = train_multi(x_multi, y_multi)


    # multi_history, multi_params = train_multi(x_multi, y_multi)

    # print('\n\n\n')
    # print('Printing Binary Classification histories')
    # print('\nParams: ',bin_params)
    # print_histories(bin_history)
    # print('\n\n')
    # print('Printing Multiclass Classification histories')
    # print('\nParams: ',multi_params)
    # print_histories(multi_history)


