import pandas as pd
import numpy as np
import time
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt



'Reading Train and Test data'

TRAINING_INPUT = "train.csv"
TEST_INPUT = "test.csv"

print 'Loading data...','\n'
start_time = time.time()

train = pd.read_csv(TRAINING_INPUT)
test =  pd.read_csv(TEST_INPUT)

print 'Loading done.', 'Time taken in loding the data is', ("--- %s seconds ---" %  str(time.time() - start_time)),'\n'

'Combine both train and test data sets into one, perform feature engineering. Lets combine them into a dataframe data with a source column specifying where each observation belongs.'

train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)


if False:
    print 'Train data has {} sample and {} features (including target label).'.format(train.shape[0],train.shape[1]), '\n'
    print 'Test data has {} sample and {} features.'.format(test.shape[0],test.shape[1]),'\n'
    print 'Data (Train + Test) data has {} sample and {} features (including target label).'.format(data.shape[0],data.shape[1]),'\n'

    'Total Missing value count Data'
    print 'Features and Lables along with total missing count', '\n'
    print data.apply(lambda x: sum(x.isnull()))


    'Total Unique Valus in each Features'
    print 'Features and Lables along with total uinque count', '\n'
    print data.apply(lambda x: len(x.unique()))





'Variable Identification'

#print data.head(5)

numerical_features = []
catagorical_columns = []
string_features = []

for column in data.columns:
    if data[column].dtype == 'int64' or data[column].dtype == 'float64' :
        numerical_features.append(column)
    else:
        data[column].dtype == 'object'
        string_features.append(column)

print 'There are total {} numerial features and {} String features in the data'.format(len(numerical_features),len(string_features)), '\n'

'Plots a scatterplot matrix of numrical featues subplots.'

if False:
    i,j = 1,10

    df = pd.DataFrame(data, columns= numerical_features[i:j])
    scatter_matrix(df, alpha=0.2, figsize=(20, 20), diagonal='kde')
    plt.show()

if False:
    'Numerial features analysis'

    print 'Descriptive Statisc on Numerial features', '\n'

    data_summary = data.describe().copy()

    medians = []
    mode = []
    for numerical_feature in numerical_features:
        medians.append(data[numerical_feature].median())
        mode.append(data[numerical_feature].mode())
    medians = pd.Series(medians, index=numerical_features)
    mode = pd.Series(mode, index=numerical_features)

    data_summary.loc['median'] = medians
    data_summary.loc['mode'] = mode

    missing_values = data.apply(lambda x: sum(x.isnull()))
    data_summary.loc['missing_value_count'] = missing_values

    total_value_count = pd.Series(data.shape[0], index=numerical_features)
    data_summary.loc['total_value_count'] = total_value_count


    unique_count = data.apply(lambda x: len(x.unique()))
    data_summary.loc['uinque_value_count'] = unique_count


    'Get Descriptive Statisc on any single Numperical Feature'

    if True:
        feature_name = 'target'
        print data_summary[feature_name]
        print(data[feature_name].unique())
        print data[feature_name].value_counts()
        print data[feature_name].values()


if False:
    'Ploting Histogram'
    feature_name = 'v2'
    plt.close('all')
    plt.figure(figsize=(15, 20))

    plt.subplot(1,2,1)
    data[feature_name].plot(kind='box', fontsize=10)
    plt.xlabel(feature_name, fontsize=15)

    plt.subplot(1,2,2)
    data[feature_name].plot(kind='hist', fontsize=10)
    plt.ylabel('Frequency', fontsize=15)

    plt.show()



'Catogorical features analysis'

print 'Descriptive Statisc on Catogorical features', '\n'
















