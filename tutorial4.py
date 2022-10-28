import botometer 
import pandas as pd 
import matplotlib.pyplot as plt 
import traceback



"""
this program uses the training set to create histograms for 
each account type (human, bot, organization) using the 
english and universal display scores. The results of running this 
should be: 
    - 6 pngs with 7 subplots each 
    - 6 csv files with 7 columns each 
"""
rapidapi_key = "2d162506e4msh29704bb041297a3p1d9925jsn7c743aef8f16" 

twitter_app_auth = { 

    'consumer_key': '6jdhW8b0KSzwk4w8rQloLVV27', 

    'consumer_secret': 'HhhWJokZFEZfX31zukGRUNi7sonZGzDaB3jUgCNGbzX7gGQKQk', 

     'access_token': '799590801101783042-tcjEAoh0GpaJvpebs4BrN2e0nN9JMIB', 

     'access_token_secret': 'BlYXrqrcmHT0cLMHLN7U7cchzQRLG9g3NVvi9J0Z92elR', 

} 

bom = botometer.Botometer(wait_on_ratelimit=True, 

                          rapidapi_key=rapidapi_key, 

                          **twitter_app_auth) 

  

# turning the id column into a list makes it easier to iterate through 

# training_set = pd.read_csv(r'/users/Mmile/Documents/twitter_accounts.csv') 
# ids = training_set['handles'].tolist()

data_path = r"D:/users/Mmile/Documents/CS"

training_set = pd.read_csv(data_path + r'/split.csv', dtype = {'id': 'str', 'type': 'str'})
training_set.drop_duplicates()
ids = training_set['id'].tolist() 


def label_conversion(type_name):
    # print(type_name["type"])
    # quit()
    if type_name == 'human':
        return 1
    elif type_name == 'ORGANIZATION':
        return 2
    else:
        return 0 # bot



# these empty dataframes will be used to hold english and universal scores for each account type 

human_eng = pd.DataFrame(columns=["id", "astroturf", "fake follower", 

                                  "financial", "other", "overall", "self-declared", "spammer", "labels"]) 

bot_eng = pd.DataFrame(columns=["id", "astroturf", "fake follower", 

                                "financial", "other", "overall", "self-declared", "spammer", "labels"]) 

org_eng = pd.DataFrame(columns=["id", "astroturf", "fake follower", 

                                "financial", "other", "overall", "self-declared", "spammer", "labels"]) 

human_univ = pd.DataFrame(columns=["id", "astroturf", "fake follower", 

                                   "financial", "other", "overall", "self-declared", "spammer", "labels"]) 

bot_univ = pd.DataFrame(columns=["id", "astroturf", "fake follower", 

                                 "financial", "other", "overall", "self-declared", "spammer", "labels"]) 

org_univ = pd.DataFrame(columns=["id", "astroturf", "fake follower", 

                                 "financial", "other", "overall", "self-declared", "spammer", "labels"]) 


counter1fail = 0
counter2success = 0

day = 1
daily_limit = 6000
start_index = (day-1) * daily_limit
end_index = min (len (ids), start_index + daily_limit)
i = start_index

for id, result in bom.check_accounts_in(ids[start_index:end_index]):
    type_name = training_set.iloc[i]['type']
    i += 1
    if 'user' not in result:
        print (str(id) + "does not exist")
        counter1fail += 1
        continue
    else:
        counter2success +=1

    # else:
    #     print (str(id) + "fetched")
    #for loop uses the list of ids to get dictionary of information for each account

    #try statement throws an exception if corresponding twitter account cannot be found for id 

  

    try: 

        if result["user"]["majority_lang"] == 'en': 

            # if the majority of the user's tweets are in english, use the english display scores 

            # otherwise, use the universal display scores 

            # creates 7x1 row dataframe that contains the account's scores 

            # concatenates this row to one of the original dataframes based on the account's type and language 

            row = pd.DataFrame([[
                id,  

                result['display_scores']['english']['astroturf'], 

                result['display_scores']['english']['fake_follower'], 

                result['display_scores']['english']['financial'], 

                result['display_scores']['english']['other'], 

                result['display_scores']['english']['overall'], 

                result['display_scores']['english']['self_declared'], 

                result['display_scores']['english']['spammer'], 
                label_conversion(type_name)]],
                columns=["id", "astroturf", "fake follower", 
                         "financial", "other", "overall", "self-declared", "spammer", "labels"]) 

  

            if (training_set.iat[ids.index(id), 1]).lower() == "human": 

                human_eng = pd.concat([human_eng, row], ignore_index=True) 

            elif (training_set.iat[ids.index(id), 1]).lower() == "bot": 

                bot_eng = pd.concat([bot_eng, row], ignore_index=True) 

            elif (training_set.iat[ids.index(id), 1]).lower() == "organization": 

                org_eng = pd.concat([org_eng, row], ignore_index=True) 

  

        else: 

            row = pd.DataFrame([[ 
                id, 

                result['display_scores']['universal']['astroturf'], 

                result['display_scores']['universal']['fake_follower'], 

                result['display_scores']['universal']['financial'], 

                result['display_scores']['universal']['other'], 

                result['display_scores']['universal']['overall'], 

                result['display_scores']['universal']['self_declared'], 

                result['display_scores']['universal']['spammer'], 
                label_conversion(type_name)]],

                columns=["id", "astroturf", "fake follower", 

                         "financial", "other", "overall", "self-declared", "spammer", "labels"]) 

  

            if (training_set.iat[ids.index(id), 1]).lower() == "human": 

                human_univ = pd.concat([human_univ, row], ignore_index=True) 

            elif (training_set.iat[ids.index(id), 1]).lower() == "bot": 

                bot_univ = pd.concat([bot_univ, row], ignore_index=True) 

            elif (training_set.iat[ids.index(id), 1]).lower() == "organization": 

                org_univ = pd.concat([org_univ, row], ignore_index=True) 

  

        # these strings make it easy to see how many ids you have gone through 

        # they are also useful for detecting errors in your code 

        print(f'{id} has been processed.') 

  

    except Exception as e: 
        traceback.print_exc()
        print("{} could not be fetched: {}".format(id, e)) 

print("failed: " + str(counter1fail))
print("success: " + str(counter2success))
  

# the contents of the dataframes are strings but histograms cant use strings 

# astype() converts the dataframes to floats 

# so that they can be used to make histograms 

human_eng = human_eng.astype({"id" :  "str", "astroturf" : float, "fake follower" : float,
                              "financial" : float, "other" : float, "overall" : float, "self-declared" : float,
                              "spammer" : float, "labels" : "int32"})
bot_eng = bot_eng.astype({"id" :  "str", "astroturf" : float, "fake follower" : float,
                              "financial" : float, "other" : float, "overall" : float, "self-declared" : float,
                              "spammer" : float, "labels" : "int32"})
org_eng = org_eng.astype({"id" :  "str", "astroturf" : float, "fake follower" : float,
                              "financial" : float, "other" : float, "overall" : float, "self-declared" : float,
                              "spammer" : float, "labels" : "int32"})
human_univ = human_univ.astype({"id" :  "str", "astroturf" : float, "fake follower" : float,
                              "financial" : float, "other" : float, "overall" : float, "self-declared" : float,
                              "spammer" : float, "labels" : "int32"})
bot_univ = bot_univ.astype({"id" :  "str", "astroturf" : float, "fake follower" : float,
                              "financial" : float, "other" : float, "overall" : float, "self-declared" : float,
                              "spammer" : float, "labels" : "int32"})
org_univ = org_univ.astype({"id" :  "str", "astroturf" : float, "fake follower" : float,
                              "financial" : float, "other" : float, "overall" : float, "self-declared" : float,
                              "spammer" : float, "labels" : "int32"})

  

# saves all the dataframes to two csv files 

# uses concat() to create a hierarchically indexed dataframe 

# one csv file for english accounts and another for universal 

# index=False keeps the dataframe from creating a blank column for the indexes 

eng_merged = pd.concat([human_eng, bot_eng, org_eng])

eng_merged.to_csv(data_path + r'set1.csv', index=False)

univ_merged = pd.concat([human_univ, bot_univ, org_univ])

univ_merged.to_csv(data_path + r'set2.csv', index=False)



# formats and saves all the histogram plots to png files 

# figsize allows you to specify the size of the graph 

# tight_layout() keeps the graphs and titles from overlapping 

# w_pad and h_pad are used to specify how much space is between each graph 

# all of these are optional. i like the way these values make the graphs look, 

# but you can change them to whatever you want 

human_eng_hist = human_eng.hist(column = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer"], figsize=(15, 12))

plt.tight_layout(w_pad=3, h_pad=3) 

plt.savefig(data_path + r'/plot_1.png')

bot_eng_hist = bot_eng.hist(column = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer"], figsize=(15, 12))

plt.tight_layout(w_pad=3, h_pad=3) 

plt.savefig(data_path + r'/plot_2.png')

org_eng_hist = org_eng.hist(column = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer"], figsize=(15, 12))

plt.tight_layout(w_pad=3, h_pad=3) 

plt.savefig(data_path + r'/plot_3.png')

human_univ_hist = human_univ.hist(column = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer"], figsize=(15, 12))

plt.tight_layout(w_pad=3, h_pad=3) 

plt.savefig(data_path + r'/plot_4.png')

bot_univ_hist = bot_univ.hist(column = ["astroturf", "fake follower", "financial", "other", "overall", "self-declared", "spammer"], figsize=(15, 12))

plt.tight_layout(w_pad=3, h_pad=3) 

plt.savefig(data_path + r'/plot_5.png')

org_univ_hist = org_univ.hist(column = ["astroturf", "fake follower", "financial", "other", "overall",
                                        "self-declared", "spammer"], figsize=(15, 12))

plt.tight_layout(w_pad=3, h_pad=3) 

plt.savefig(data_path + r'/plot_6.png')

  

# displays the histograms. if you dont care to see them then you dont have to include this 

# they will be saved to the png files regardless 

#failed: 363
#success: 5189

plt.show()
