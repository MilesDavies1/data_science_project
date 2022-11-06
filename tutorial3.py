import botometer
import pickle
import pprint
import pandas
import traceback

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

# Check a sequence of accounts
df = pandas.read_csv(r'/users/Mmile/Documents/twitter_accounts.csv')
accounts = df['handles'].tolist()
print(accounts)

account_list = []
account_index = []

for screen_name, result in bom.check_accounts_in(accounts):
    # this will be appended to the new dataframe
    row = {}
    def label_conversion(row):      
        if row["type"] == 'human':    
            return 1    
        elif row["type"] == 'ORGANIZATION':    
            return 2    
        else:    
            return 0 # bot 

    # we use a try-catch because we do not want it to stop execution if botometer fails to get stats on an account.
    try:
        if result["user"]["majority_lang"] == 'en':
            row = {
                "id": result["user"]["user_data"]["id_str"],
                "CAP": result['cap']['english'],
                "astroturf": result['display_scores']['english']['astroturf'],
                "fake_follower": result['display_scores']['english']['fake_follower'],
                "financial": result['display_scores']['english']['financial'],
                "other": result['display_scores']['english']['other'],
                "overall": result['display_scores']['english']['overall'],
                "self-declared": result['display_scores']['english']['self_declared'],
                "spammer": result['display_scores']['english']['spammer'],
                #"type": result['display_scores']['english']['type'],
            }
        else:
            row = {
                "id": result["user"]["user_data"]["id_str"],
                "CAP": result['cap']['universal'],
                "astroturf": result['display_scores']['universal']['astroturf'],
                "fake_follower": result['display_scores']['universal']['fake_follower'],
                "financial": result['display_scores']['universal']['financial'],
                "other": result['display_scores']['universal']['other'],
                "overall": result['display_scores']['universal']['overall'],
                "self-declared": result['display_scores']['universal']['self_declared'],
                "spammer": result['display_scores']['universal']['spammer'],
                #"type": result['display_scores']['universal']['type']
            }

        account_list.append(row)
        account_index.append(screen_name)

        # notify that we are done processing
        print(f'{result["user"]["user_data"]["id_str"]} has been processed.')

    # skip if error
    except Exception as e:
        traceback.print_exc()
        print("{} Could not be fetched: {}".format(id, e))

account_info_df = pandas.DataFrame(account_list, index=account_index)

account_info_df.to_csv('api_account_info.csv') # you can name the file whatever you want
  
