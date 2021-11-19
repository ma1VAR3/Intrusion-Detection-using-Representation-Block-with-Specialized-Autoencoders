from numpy.lib.function_base import select


def encodeCategorical(df):
    from sklearn.preprocessing import LabelEncoder
    
    le = LabelEncoder()

    df['labels'] = le.fit_transform(df['labels'])
    df['protocol_type'] = le.fit_transform(df['protocol_type'])
    df['service'] = le.fit_transform(df['service'])
    df['flag'] = le.fit_transform(df['flag'])

    return df

def scaleData(x):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    sc = StandardScaler()
    x = sc.fit_transform(x)
    x = pd.DataFrame(x)

    return x

def reduceFeaturespace(X_train, y_train, feature_dim, mode):
    
    import numpy as np
    if mode == 'dtc':
        print('DTC')
        from sklearn.feature_selection import RFE
        from sklearn.tree import DecisionTreeClassifier
        m = DecisionTreeClassifier()
        selector = RFE(m, n_features_to_select= feature_dim).fit(X_train,y_train)
        indices = np.where(selector.support_==True)[0]
        features = X_train.columns.values[indices]
        X_train= X_train[features]
        return X_train, features
    elif mode == 'svc':
        print('SVC')
        from sklearn.feature_selection import RFE
        from sklearn.svm import SVC
        m = SVC(kernel="poly")
        selector = RFE(m, n_features_to_select= feature_dim).fit(X_train,y_train)
        indices = np.where(selector.support_==True)[0]
        features = X_train.columns.values[indices]
        X_train= X_train[features]
        return X_train, features
    elif mode == 'rf':
        print('RF')
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier()
        selector = RFE(m, n_features_to_select= feature_dim).fit(X_train,y_train)
        indices = np.where(selector.support_==True)[0]
        features = X_train.columns.values[indices]
        X_train= X_train[features]
        return X_train, features
    elif mode == 'chi2':
        print('CHI2')
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        selector = SelectKBest(chi2, k= feature_dim).fit(X_train,y_train)
        indices = np.where(selector.support_==True)[0]
        features = X_train.columns.values[indices]
        X_train= X_train[features]
        return X_train, features
    else:
        print('DTC')
        from sklearn.feature_selection import RFE
        from sklearn.tree import DecisionTreeClassifier
        m = DecisionTreeClassifier()
        selector = RFE(m, n_features_to_select= feature_dim).fit(X_train,y_train)
        indices = np.where(selector.support_==True)[0]
        features = X_train.columns.values[indices]
        X_train= X_train[features]
        return X_train, features
    