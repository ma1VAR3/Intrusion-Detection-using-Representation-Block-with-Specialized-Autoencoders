from scipy.sparse import data


def getdata(feature_dim):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import keras
    from keras.utils import np_utils

    from preprocessing import encodeCategorical, scaleData, reduceFeaturespace

    df = pd.read_csv('./datasets/NSLKDD/KDDTrain+.txt')
    cols = get_cols()
    df.columns = cols

    df = encodeCategorical(df)
    x = df.drop('labels', axis=1)
    y = df.loc[:, ['labels']]

    X_train = scaleData(x)

    y_train = np_utils.to_categorical(y)

    X_train = reduceFeaturespace(X_train, y_train, feature_dim, 'dtc')

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def getattackdata(feature_dim, dataset):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import keras
    from keras.utils import np_utils

    from preprocessing import encodeCategorical, scaleData, reduceFeaturespace

    
    csv = './datasets/NSLKDD/KDDTrain+.txt'
    
    df = pd.read_csv(csv)
    
    
    cols = get_cols()
    df.columns = cols

    # categories = get_labels('priviledge')
    # df['drop'] = df.apply(lambda x: 1 if (x['labels'] in categories or x['labels']=='normal') else 0, axis=1)
    
    df['drop'] = df.apply(lambda x: 1 if (x['labels']=='normal') else 0, axis=1)
    idx_p = np.where(df['drop']==1)[0]
    df = df.drop(idx_p)

    df = encodeCategorical(df)
    x = df.drop('labels', axis=1)
    x = x.drop('level', axis=1)
    x = x.drop('drop', axis=1)
    y = df.loc[:, ['labels']]

    X_train = scaleData(x)

    y_train = np_utils.to_categorical(y)

    # X_train = X_train[feats]
    X_train, feats = reduceFeaturespace(X_train, y_train, feature_dim, 'dtc')

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, feats

def getbinarydata(feature_dim, dataset):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from preprocessing import encodeCategorical, scaleData, reduceFeaturespace
    import pickle

    
    csv = './datasets/NSLKDD/KDDTrain+.txt'
    
    df = pd.read_csv(csv)
    
    cols = get_cols()
    df.columns = cols
    
    df['is_attacked'] = df.apply(lambda x: 0 if x['labels']=='normal' else 1, axis=1)
    
    

    attacked_df = df.copy()
    benign_df = df.copy()

    # benign_df['drop'] = benign_df.apply(lambda x: 1 if (x['is_attacked'] == 1 or x['labels'] in categories) else 0, axis=1)
    benign_df['drop'] = benign_df.apply(lambda x: 1 if (x['is_attacked'] == 1) else 0, axis=1)
    idx_b = np.where(benign_df['drop']==1)[0]
    benign_df = benign_df.drop(idx_b)

    # attacked_df['drop'] = attacked_df.apply(lambda x: 1 if (x['is_attacked'] == 0 or x['labels'] in categories) else 0, axis=1)
    attacked_df['drop'] = attacked_df.apply(lambda x: 1 if (x['is_attacked'] == 0) else 0, axis=1)
    idx_a = np.where(attacked_df['drop']==1)[0]
    attacked_df = attacked_df.drop(idx_a)

    # df['drop'] = df.apply(lambda x: 1 if (x['labels'] in categories) else 0, axis=1)
    # idx_p = np.where(df['drop']==1)[0]
    # df = df.drop(idx_p)

    
    benign_df= encodeCategorical(benign_df)
    attacked_df= encodeCategorical(attacked_df)
    df = encodeCategorical(df)
    

    x = df.drop('is_attacked', axis=1)
    x = x.drop('labels', axis=1)
    x = x.drop('level', axis=1)

    # x = x.drop('drop', axis=1)
    y = df.loc[:, ['is_attacked']]
    x = scaleData(x)
    x, feats = reduceFeaturespace(x, y, feature_dim, 'dtc')

    
    x_b = benign_df.drop('is_attacked', axis=1)
    print("xb shape: ", x_b.shape)
    x_b = x_b.drop('labels', axis=1)
    x_b = x_b.drop('level', axis=1)
    x_b = x_b.drop('drop', axis=1)
    y_b = benign_df.loc[:, ['is_attacked']]
    x_b = scaleData(x_b)
    x_b = x_b[feats]

    
    x_a = attacked_df.drop('is_attacked', axis=1)
    x_a = x_a.drop('labels', axis=1)
    x_a = x_a.drop('level', axis=1)
    x_a = x_a.drop('drop', axis=1)
    y_a = attacked_df.loc[:, ['is_attacked']]
    x_a = scaleData(x_a)
    x_a = x_a[feats]

    with open('rfe_binary.pkl','wb') as rfeb:
        pickle.dump(feats, rfeb, pickle.HIGHEST_PROTOCOL)

    return x_b, y_b, x_a, y_a, x, y, feats

def getcategorydata(feature_dim, dataset, feats):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import keras
    from keras.utils import np_utils
    from preprocessing import encodeCategorical, scaleData, reduceFeaturespace
    
    x_all = []
    y_all = []
    cats = get_categories(dataset)
    
    for c in cats:
        print("Category: ", c)
        df = pd.read_csv('./datasets/NSLKDD/KDDTrain+.txt')
        cols = get_cols()
        df.columns = cols
        
        categories = get_labels(c)
        df['drop'] = df.apply(lambda x: 0 if x['labels'] in categories else 1, axis=1)
        idx = np.where(df['drop']==1)[0]
        df = df.drop(idx)
        print(df.labels.unique())
        
        df= encodeCategorical(df)
        x = df.drop('labels', axis=1)
        x = x.drop('level', axis=1)
        x = x.drop('drop', axis=1)
        y = df.loc[:, ['labels']]
        x = scaleData(x)
        x = x[feats]

        x_all.append(x)
        y_all.append(y)

    return x_all, y_all

def print_histories(histories):
    for i in range(len(histories)):
        print('-'*15, '>', f'Fold {i+1}', '<', '-'*15)
        print(histories[i])


def get_labels(attack_class):
    dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
    probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
    privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
    access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']
    
    
    if attack_class == 'dos':
        return dos_attacks
    if attack_class == 'probe':
        return probe_attacks
    if attack_class == 'priviledge':
        return privilege_attacks
    if attack_class == 'access':
        return access_attacks
    

def get_categories(dataset):
    if dataset == "nslkdd":
        return ['dos', 'probe', 'access', 'priviledge']
    

def get_cols():
    columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes'
    ,'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised'
    ,'root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files'
    ,'num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
    ,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate'
    ,'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate'
    ,'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate'
    ,'dst_host_rerror_rate','dst_host_srv_rerror_rate','labels','level'])
    
    return columns