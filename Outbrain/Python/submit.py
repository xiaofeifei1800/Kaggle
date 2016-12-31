import pandas

csv_delimiter = ' '

print "load"
df = pandas.read_csv("/Users/xiaofeifei/GitHub/Kaggle/Outbrain/Python/sub_proba.csv")

print "sort"
df.sort_values(['display_id','clicked'], inplace=True, ascending=False)

print "combine"
subm = df.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()

print "write"
subm.to_csv("subm.csv", index=False)
