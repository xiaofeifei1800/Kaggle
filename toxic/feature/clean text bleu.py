import pandas as pd

train = pd.read_csv('/Users/guoxinli/I/data/data/clean_train_v2.csv', nrows=1000).fillna(' ')
test = pd.read_csv('/Users/guoxinli/I/data/data/clean_test_v2.csv', nrows=1000).fillna(' ')


def bleu(c,r,iwords= set(),iwords2 = set()):  # function to calc the simiarlity of two words
    if len(c) == 0 or len(r) == 0:
        return 0.0
    if len(set(r) & set(c)) < len(set(r)) * 0.7:
        return 0.0
    if len(c) > len(r):
        bp = 0
    else:
        bp = max(min(1-len(r) * 1.0/len(c),0),-0.5)
    sumpn = 0.0
    for i in range(3):
        rcount = {}
        for j in range(len(r) - i):
            w = 1
            rcount[" ".join(r[j:j+1+i])] = rcount.get(" ".join(r[j:j+1+i]),0) + w
        ccount = {}
        ctcount = {}
        for j in range(len(c) - i):
            w = 1
            t1 = " ".join(c[j:j+1+i])
            ccount[t1] = min(ccount.get(t1,0) + w,rcount.get(t1,0))
            ctcount[t1] = ctcount.get(t1, 0) + w
        temp = 0.33 * sum(map(lambda x:x[1],ccount.items()))*1.0/(sum(map(lambda x:x[1],ctcount.items()))+0.0001)
        if temp < 0.25:
            return 0.0
        sumpn += temp
    return bp + sumpn


wordcount = {}
for e in train.comment_text:
    for word in e.split(' ',-1):
        word = word.lower()
        if len(word) == 0:
            continue
        wordcount[word] = wordcount.get(word,0.0) + 1.0


r = len(train)*0.5/len(test)
print(r)
wordcount2 = {}
for e in test.comment_text:
    for word in e.split(' ',-1):
        word = word.lower()
        if len(word) == 0:
            continue
        wordcount2[word] = wordcount2.get(word,0.0) + r



sortcount = sorted(wordcount.items(),key=lambda x:-x[1])
wordindex = {}
validword = {}
index4 = 2
count = 0
abc = set(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])

for k,v in sortcount:
    abstart = k[0] in abc
    isfind = False
    k2 = k.replace("'","")
    scoremax = 0.0
    if round(v) > 10 and len(k) > 4:
        print(k)
        print(v)
        taglist = []
        taglist.extend(wordindex.get(k[:3],[]))
        if not abstart:
            taglist.extend(wordindex.get(k[1:4],[]))
        nearword = ""
        for word in taglist:
            score = 0
            if abs(len(k) - len(word)) > 3:
                continue
            score += bleu(k2, word)
            if score > 0.80:
                scoremax = score
                nearword = word
                isfind = True
                break
            if isfind:
                validword[k] = validword[nearword]
            elif (round(v) > 1 or round(wordcount2.get(k, 0)) > 1) and len(k) > 4:
                scoremax = 0.0
                taglist = []
                taglist.extend(wordindex.get(k[:3], []))
                if not abstart:
                    taglist.extend(wordindex.get(k[1:4], []))
                nearword = ""
                for word in taglist:
                    if wordcount.get(word) <= 10:
                        continue
                    if abs(len(k) - len(word)) > 3:
                        continue
                    score = 0

                    score += bleu(k2, word)
                    if score > max(0.60, scoremax):
                        scoremax = score
                        # print word,k,score
                        nearword = word
                        isfind = True
                        if score > 0.75:
                            break
                if isfind:
                    validword[k] = validword[nearword]
            if not isfind and round(v) >= 7:
                validword[k] = [index4]
                index4 += 1
                wordindex[k[:3]] = wordindex.get(k[:3], [])
                wordindex[k[:3]].append(k)
            elif isfind and scoremax > 0.70:
                wordindex[k[:3]] = wordindex.get(k[:3], [])
                wordindex[k[:3]].append(k)
            count += 1
            if count % 10000 == 0:
                print(count)

print(wordcount)
print(validword)
print(wordindex)