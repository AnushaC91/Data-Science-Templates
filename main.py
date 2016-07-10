
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import Imputer
import xgboost as xgb
from sklearn.base import TransformerMixin
from sklearn.metrics import log_loss


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


if __name__=='__main__':

    'Reading Train and Test data'

    TRAINING_INPUT = "train.csv"
    TEST_INPUT = "test.csv"

    print 'Loading data...'

    train = pd.read_csv(TRAINING_INPUT)
    test =  pd.read_csv(TEST_INPUT)
    '''

    'Exploring'

    # print train.columns
    # print train.apply(lambda x: len(x.unique()))

    'Filter categorical variables'

    print 'Exploring data...'

    categorical_columns = [x for x in train.dtypes.index if train.dtypes[x] == 'object']

    # Print frequency of categories
    for col in categorical_columns:
        print '\nFrequency of Categories for variable %s' %col
        print train[col].value_counts()

    '''

    'Cleaning Data'

    print "Cleaning data..."

    # Imputing Missing values

    train = DataFrameImputer().fit_transform(train)
    test = DataFrameImputer().fit_transform(test)

    print train.shape
    print test.shape

    train_final = train.drop(["ID", "v22"], axis=1, inplace=False)
    # df_final = train_final
    out_id = test["ID"].as_matrix()

    X_test = test.drop(["ID", "v22"], axis=1, inplace=False)

    y_train = train_final['target'].as_matrix()
    X_train = train_final.drop('target', axis=1, inplace = False)

    'Vectorizing'

    print "Vectorizing..."

    cols_to_retain = ['v107', 'v110', 'v112', 'v113', 'v125', 'v24', 'v3', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 'v79', 'v91']

    cat_train = X_train[cols_to_retain]
    cat_test = X_test[cols_to_retain]

    x_num_train = X_train.drop(cols_to_retain, axis=1, inplace=False).as_matrix()
    x_num_test = X_test.drop(cols_to_retain, axis=1, inplace=False).as_matrix()

    x_cat_train = cat_train.T.to_dict().values()
    x_cat_test = cat_test.T.to_dict().values()

    # vectorize

    vectorizer = DV(sparse=False)
    vec_x_cat_train = vectorizer.fit_transform(x_cat_train)
    vec_x_cat_test = vectorizer.transform(x_cat_test)

    # complete x

    X_train = np.hstack((x_num_train, vec_x_cat_train))
    X_test = np.hstack((x_num_test, vec_x_cat_test))

    print X_train.shape
    print X_test.shape



    ' Modelling '
    '''

    clfRF = RandomForestClassifier(n_estimators=300, min_samples_split=4, max_features=80, n_jobs=-1, oob_score=True)

    print 'Modelling using RF Classifier :', clfRF
    print "Training using RF started..."

    clfRF.fit(X_train, y_train)
    y_predicted = clfRF.predict(X_train)
    y_predict_prob = clfRF.predict_proba(X_train)

    # print cross_validation.cross_val_score(clfRF, X_train,y_train, cv=5, scoring='log_loss')
    # print 'OOB Score :', clfRF.oob_score_
    print 'RF feature importance:', clfRF.feature_importances_
    print "F-1 score of the RF model on training data set is ", f1_score(y_train, y_predicted)
    print "log-loss of the RFC model on training data set is ", log_loss(y_train, y_predict_prob)

    '''

    # print "Modelling GBC"
    # print "Training using GBC started..."
    # clf = GradientBoostingClassifier(n_estimators=300, min_samples_leaf=50, min_samples_split=50, max_depth=9)
    # clf.fit(X_train, y_train)
    # print 'GBDT Complete', clf
    # print 'GBDT :', sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), df_final.columns),reverse=True)
    # print "F-1-Score of the GBC model on training data set is ", f1_score(y_predicted, y_train)


    print "Modelling using XGBT"

    params = {}
    params["objective"] = "binary:logistic"
    # params["eta"] = 0.005
    params["min_child_weight"] = 6
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.6
    # params["scale_pos_weight"] = 0.8
    params["silent"] = 1
    params["max_depth"] = 8
    params["max_delta_step"]=2
    params["seed"] = 0
    num_rounds = 500

    # Training

    print 'XGBoost Training Process Started'

    xgtrain = xgb.DMatrix(X_train, y_train)
    xgtest = xgb.DMatrix(X_test)



    plst = list(params.items())
    model = xgb.train(plst, xgtrain, num_rounds)
    y_predict_prob = model.predict(xgtrain)

    print 'XGB Model trained :', model
    # print "F-1 score of the RF model on training data set is ", f1_score(y_train, y_predicted)
    # print "log-loss of the XGB model on training data set is ", log_loss(y_train, y_predict_prob)

    predictions = model.predict(xgtest)
    print 'Predictions Complete...'
    out_df = pd.DataFrame({"ID":out_id, "PredictedProb": predictions})
    out_df.to_csv("XGBT_submission.csv", index=False, header=True)


