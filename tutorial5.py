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
  df = df.replace({'labels': 2}, {'labels': 1})
  #df[df['label'] == 2] = 0  
  return df

def fix_positive_condition(df):
  df = df.replace({'labels': 1}, {'labels': 3})
  df = df.replace({'labels': 0}, {'labels': 1})
  df = df.replace({'labels': 3}, {'labels': 0})
  return df
  
def logisticRegression(path):
	
	master_df = pd.read_csv(path)
	
	#df = turn_orgs_to_bots(master_df)
	# features 
	x1 = master_df.drop(['labels'], axis=1).values

	# labels
	y1 = master_df['labels'].values
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100)

	X_train_scaled = preprocessing.scale(X_train)
	X_test_scaled = preprocessing.scale(X_test)
	model = LogisticRegression()
	model.fit(X_train_scaled, Y_train)
	result = model.score(X_test_scaled, Y_test)
	return result
#print("Accuracy: %.2f%%" % (result * 100.0))

def remove_outliers(zscore, pathA, pathB):  
  mdf = pd.read_csv(pathA)
  mdf = fix_positive_condition(mdf)  
  mdf = turn_orgs_to_bots(mdf)
  columns = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer", "labels"]  
  mdf= mdf[columns]
    # separate them by category  
  human_df = mdf[mdf["labels"] == 1]  
  non_human_df = mdf[mdf["labels"] == 0]  
  columns = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer"]
    # filter out the outliers  
    # CAP,astroturf,fake_follower,financial,other,overall,self-declared  
    #columns = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer"]  
    #for humans in columns : ["astroturf", "fake follower", "financial", "other", "overall", "self-declared"]  
    #columns_list = ["astroturf", "overall", "spammer",]
  human_df = human_df[np.abs(stats.zscore(human_df[columns]) < zscore).all(axis=1)]  
  
    # for non_humans  
  non_human_df = non_human_df[np.abs(stats.zscore(non_human_df[columns]) < zscore).all(axis=1)]  
  
    #print(human_df.describe())  
    #print(non_human_df.describe())  
  
    # combine both of the dataframe and export  
  master = pd.concat([human_df, non_human_df])  
  
    #print(master.describe())  
  master.to_csv(pathB, index=False)

def confusion_matrix(path, lang):
  master_df = pd.read_csv(path)
  x1 = master_df.drop(['labels'], axis=1).values
  y1 = master_df['labels'].values

  X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x1, y1, test_size=0.30, random_state=100)

  X_train_scaled = preprocessing.scale(X_train)
  X_test_scaled = preprocessing.scale(X_test)
  model = LogisticRegression()
  model.fit(X_train_scaled, Y_train)

  title_options = [("Confusion Matrix, without normalization", None), ("Normalization: true", "true"), 
                    ("Normalization: pred", "pred"), ("Normalization: all", "all")]
  for title, normalize in title_options:
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, Y_test, display_labels=["bot", "human"], 
                                          cmap=plt.cm.Blues, normalize=normalize)
    file_path = r'D:/users/Mmile/Documents/CS/file-{}-{}.png'.format(normalize, lang)
    disp.ax_.set_title(title)
    plt.savefig(file_path)


#removes the specified amount of outliers/lareger the z score, the further away the account has to be from the mean to be considered an outlier
#objective is to find the optimal zscore to find how accurately the algorithm discerns wether the account is a bot or human
#find which zscore(standard deviation) results in the highest accuracy 

zscore = 2.8 
zscore_list = []
results_list = []
while zscore >= 0:
  zscore_list.append(zscore)
  remove_outliers(zscore, universal, universal_no_outliers)
  result = logisticRegression(universal_no_outliers)
  result *= 100
  results_list.append(float("{:.2f}".format(result)))
  zscore -= 0.1

zscore_total = 0
accuracies_total = 0
for i in zscore_list:
  zscore_total += 1
for i in results_list:
  accuracies_total += 1
#print(zscore_total, accuracies_total)
#print(zscore_list, results_list)

#find the greatest value (highest accuracy) in the results_list
max_accuracy = results_list[0]
for number in results_list:
  if number > max_accuracy:
    max_accuracy = number
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
plt.xlabel('standard deviation')
plt.ylabel('accurary %')
plt.show()

# remove_outliers(best_zscore, english, english_no_outliers)
# logisticRegression(english_no_outliers)
# # confusion_matrix(english_no_outliers, "eng")

# remove_outliers(best_zscore, universal, universal_no_outliers)
# logisticRegression(universal_no_outliers)
# confusion_matrix(universal_no_outliers, "univ")