import pandas as pd 

results = pd.read_csv(r'F:\FYP\Other Datasets\densenet121.csv')


results = results.to_numpy()
print(results[29])

