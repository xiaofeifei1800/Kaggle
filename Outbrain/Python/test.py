import pandas

csv_delimiter = ' '

df = pandas.read_csv("/Users/xiaofeifei/GitHub/Kaggle/Outbrain/Python/sub_proba.csv")


df.sort_values(['display_id','clicked'], inplace=True, ascending=False)
subm = df.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()
subm.to_csv("subm.csv", index=False)
