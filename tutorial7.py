import scipy
import pickle
import xgboost
import sklearn
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import model_selection, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score

universal_no_outliers = r'D:/users/Mmile/Documents/CS/universal_no_outliers.csv'
universal = r'D:/users/Mmile/Documents/CS/universal_combined_training_set_final.csv'
english_no_outliers = r'D:/users/Mmile/Documents/CS/english_no_outliers.csv'
english = r'D:/users/Mmile/Documents/CS/english_combined_training_set_final.csv'

def turn_orgs_to_bots(df):
    df[df['labels'] == 2] = 0
    return df


def forest(path):
    master_df = pd.read_csv(path)

    removed = ['labels', 'financial', 'self-declared', 'fake_follower', 'CAP']
    x1 = master_df.drop(['labels'], axis=1).values
    # return
    y1 = master_df['labels'].values

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100,
                                                                        stratify=y1)
    X_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    model = xgboost.XGBClassifier(use_label_encoder=False)

    optimization_dict = {
        'max_depth': [2, 4, 6],
        'n_estimators': [50, 200, 500],
        'learning_rate': [0.1, 0.01, 1],
    }

    model = model_selection.GridSearchCV(model, optimization_dict,
                                         scoring='accuracy', verbose=1)
    model.fit(X_scaled, Y_train)
    result = model.score(X_test_scaled, Y_test)
    # print("Accuracy: %.2f%%" % (result * 100.0))
    # print(model.best_score_)
    # print(model.best_params_)
    return result

def remove_outliers(zscore):
    #mdf = pd.read_csv(r'C:\Users\lexyl\PycharmProjects\pythonProject\english_combined_traning_set_final.csv')
    mdf = pd.read_csv(english)
    mdf = turn_orgs_to_bots(mdf)
    columns = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer", "labels"]
    mdf = mdf[columns]

    # separate then by category
    human_df = mdf[mdf["labels"] == 1]
    bot_df = mdf[mdf["labels"] == 0]

    # filter out the outliers
    # CAP, astroturf, fake_follower, financial, other, overall, self-indicted
    # for humans
    columns = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer"]
    human_df = human_df[np.abs(stats.zscore(human_df[columns]) < zscore).all(axis=1)]
    bot_df = bot_df[np.abs(stats.zscore(bot_df[columns]) < zscore).all(axis=1)]

# print(human_df.describe())
#    print(human_df.describe())
#    print(bot_df.describe())

# combine both of the dataframe and export
    master = pd.concat([human_df, bot_df])

# print(master.describe())
    #master.to_csv(r'C:\Users\lexyl\PycharmProjects\pythonProject\english_no_outliers.csv', index=False)
    master.to_csv(english_no_outliers, index=False)

    pass

#change std
zscore = 2.5
zscore_list = []
results_list = []
while zscore >= 0:
    zscore_list.append(zscore)
    remove_outliers(zscore)
    result = forest(english_no_outliers)
    result *= 100
    results_list.append(float("{:.2f}".format(result)))
#    print(f"{std_dev} processed")
    zscore -= 1.0
zscore_total = 0 #set zscore to 0
accuracy_total = 0 #set accuracy to 0
for i in zscore_list: #start a for loop
    zscore_total += 1

for i in results_list:
    accuracy_total += 1
#print(counter_3, counter_4)
#print(zscore_list, results_list)

#find the greatest value (highest accuracy) in results_list
max_accuracy = results_list[0]
for n in results_list:
    if n > max_accuracy:
        max_accuracy = n
#print(max_accuracy)

#find the highest accuracy index and print the corresponding zscore index
counter_1 = 0
counter_2 = 0
for i in results_list:
    if i == max_accuracy:
         counter_2 = counter_1
    counter_1 += 1
best_zscore = zscore_list[counter_2]
#print(best_zscore)


plt.plot(zscore_list, results_list)
plt.xlabel('zscore')
plt.ylabel('accuracy %')

#find the highest




#plt.plot(std_dev_list, results_list)
#plt.xlabel('z score')
#plt.ylabel('accuracy %')

# universal
#remove_outliers(1.194)
# english
#remove_outliers(0.895)
#logistic_regre()
plt.show()
