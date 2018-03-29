import re
import pandas as pd
from autocorrect import spell
from string import punctuation

F = open('/Users/guoxinli/I/data/data/spell_correct.txt','r')
df = pd.read_csv("/Users/guoxinli/I/data/data/test.csv")

spell_check = {}
for line in F:
    wrong, right = line.rstrip().split('->')
    spell_check[wrong]=right

def isEnglish(s):
    try:
        text = s
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return ''
    else:
        return s

patterns = {
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8\)": " good ",
    ":-\)": " good ",
    ":\)": " good ",
    ";\)": " good ",
    "\(-:": " good ",
    "\(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":'\)": " sad ",
    ":-\(": " bad ",
    ":\(": " bad ",
    '- :':'',
    ":s": " bad ",
    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8\)": " smile ",
    ":-\)": " smile ",
    ":\)": " smile ",
    ";\)": " smile ",
    "\(-:": " smile ",
    "\(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":'\)": " sad ",
    ":-\(": " sad ",
    ":\(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"(A|a)in\'t": "is not",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    r"\bI'm\b": "i am",
    r"\'s": "is",
    r"(\w+)\'d":"\g<1> would",
    r"(\w+)\'ll": r"\g<1> will",
    r"(\w+)\'ve": r"\g<1> have",
    r"([a-z])\1{2,}": r"\1",
    r"\'re": " are",
    r"\(UTC\) ":'',
    r"fuckin": 'fucking',
    r"bithc":'bitch',
    r'wtf':"what the fuck",
    r"sUUck": 'suck',
    r"cck": 'cock',
    r"fckers":'fuckers',
    " r ": "are",
    " u ": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'll" : "i will",
    "it's" : "it is",
    "that's" : "that is",
    "weren't" : "were not",
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}": '',
    r"\[\[.*\]":"",
    r"_": '',
    r"\d+": '',
    r"\s+": ' '
}

keys = [i for i in patterns.keys()]

def text_to_wordlist(text):

    # regep
    for (pattern, repl) in patterns.items():
        text = re.sub(pattern, repl, text)

    # Misspelling
    text = text.split()
    text = [w if w not in spell_check else spell_check[w] for w in text]
    text = [isEnglish(w) for w in text]
    text = " ".join(text)

    # duplicates
    try:
        unique_len = len(set(str(text).split()))
        total_len = len(str(text).split())

        if unique_len/total_len <= 0.06:
            text = text.split()
            text = sorted(set(text), key=text.index)
            text = " ".join(text)

    except ZeroDivisionError as e:
        print(e)

    text = ''.join([c for c in text if c not in punctuation])
    text = text.lower()
    return text

df.comment_text = df.comment_text.apply(lambda x: text_to_wordlist(x))

df.to_csv("/Users/guoxinli/I/data/data/clean_test_v2.csv", index=False)