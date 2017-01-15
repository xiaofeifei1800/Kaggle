import marshal

f = open("/Users/xiaofeifei/I/Kaggle/Outbrain/all_train.csv")
# f = open("/Users/xiaofeifei/I/Kaggle/train")
fc = open("../fc","w")

d = {}
count = 0
line = f.readline()
while True:
    line = f.readline()
    if not line:
        break
    if count > 10:
        break
    count += 1
    if count % 100000 == 0:
        print count
    print line

    lis = line[:].split(",")

    for i in xrange(3,len(lis)-1):
        name = chr(ord('a') + i - 3)
        feat = name + "_" + lis[i]
        if feat in d:
            d[feat] += 1
        else:
            d[feat] = 1

s = []
dd = {}
for x in d:
    if d[x] >= 10:
        s.append(x)
marshal.dump(set(s),fc)