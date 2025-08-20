import pandas as pd
import pickle
# converts pkl file to csv to view
with open(r"C:\Projects\Lip_Reading\GRID\word_map.pkl", "rb") as f:
    word_mapf = pickle.load(f)
df = pd.DataFrame(list(word_mapf.items()), columns=["key", "value"])

df.to_csv(r"C:\Projects\Lip_Reading\GRID\word_map.csv", index=False)