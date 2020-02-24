import pandas as pd
import os
import re
import pkuseg
from tqdm import tqdm
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Data Cleansing
os.chdir(r"E:\Learning\Python\NLP related\Vocabulary")
table = pd.read_excel("10jqka_news_20190114.xlsx")
f = []

for (_, _, filenames) in os.walk(str(os.getcwd())):
    f.extend(filenames)
    break

for item in f:
    lst = re.split("\.", item)
    if lst[1] == "xlsx" and lst[0] != "10jqka_news_20190114":
        tmptable = pd.read_excel(item)
        table = table.append(tmptable)

# writer = pd.ExcelWriter('infoset.xlsx', engine = 'xlsxwriter')
# table.to_excel(writer, sheet_name='Sheet1')
# writer.save()
#
# extracted = table[['content']]
# writer = pd.ExcelWriter('extracted.xlsx', engine = 'xlsxwriter')
# extracted.to_excel(writer, sheet_name = 'Sheet1')
# writer.save()

# Sentence Segmentation
mark = []
for i in range(len(lst)):
    if type(lst[i]) != str and lst[i] not in mark:
        mark.append(lst[i])

taglist = ['n', 's', 'v', 'z', 'l', 'j', 'ns', 'nt', 'nx', 'nz', 'vd', 'vn', 'vx', 'ad', 'an']
seg = pkuseg.pkuseg(model_name = "news", postag = True)
with open("stopper.txt", mode = 'r', encoding = "UTF-8") as file:
    stop_words = file.readlines() # Use HIT stop words list
stop_words = [item.rstrip('\n') for item in stop_words]

wordlist = []
for i in range(len(lst)):
    if lst[i] in mark:
        continue
    sentence = re.sub(r'[^\u4e00-\u9fa5]', ' ', lst[i])
    text = seg.cut(sentence)
    tmplist = []
    for item in text:
        if item[1] in taglist and item[0] not in stop_words:
            tmplist.append(item[0])
    wordlist.append(tmplist)

# Calculate TF-IDF value
corpus = [' '.join(item) for item in wordlist]

vectorizer=CountVectorizer()
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
word=vectorizer.get_feature_names()

tf_idf_list = []
lstlen = len(wordlist)
with tqdm(total=lstlen) as pbar:
    pbar.set_description("Extracting tf-idf")
    for i in range(lstlen):
        pbar.update(1)
        itemset = set(wordlist[i])
        tf_idf_array = list(tfidf[i].toarray()[0])
        tf_idf_dic = {}
        for w in itemset:
            if len(w) > 1:
                tf_idf_dic[w] = tf_idf_array[word.index(w)]
        tf_idf_list.append(tf_idf_dic)

# Extract Keywords
keywordset = []
with tqdm(total=lstlen) as pbar:
    pbar.set_description("Extracting Keywords")
    for dic in tf_idf_list:
        pbar.update(1)
        tmplst = [v for v in sorted(dic.items(), key=lambda d: d[1])]
        length = len(tmplst)
        if length <= 4:
            for item in tmplst:
                keywordset.append(item[0])
        else:
            keywordset.append(tmplst[0][0])
            keywordset.append(tmplst[1][0])
            keywordset.append(tmplst[length-2][0])
            keywordset.append(tmplst[length-1][0])
keywordset = set(keywordset)

# Verify the requirement.

if "新经济" in keywordset and "粤港澳大湾区" in keywordset:
    print("Succeeded.")
else:
    print("Failed.")

# with open("Vocabulary.txt", mode = "w+") as f:
#     f.write(' '.join(keywordset))
#
# TF-IDF begins here. Extremely slow . I need to change a way of implementing it.
#
# uniquelst = []
# for item in wordlist:
#    uniquelst.extend(item)
# len(set(uniquelst))
#
# def tf(text):
#     wordtf = {}
#     wordset = set(text)
#     length = len(text)
#     for word in wordset:
#         wordcnt = 0
#         for item in text:
#             if item == word:
#                 wordcnt += 1
#         wordtf[word] = wordcnt/length
#     return wordtf
#
# def idf(wordlist):
#     tmplst =[]
#     for item in wordlist:
#         tmplst.extend(item)
#     wordset = set(tmplst)
#     length = len(wordset)
#     wordidf = {}
#     wordcnt = 0
#     cnt = 0
#     for word in wordset:
#         cnt += 1
#         print(cnt)
#         for tmpword in tmplst:
#             if word == tmpword:
#                 wordcnt += 1
#         wordidf[word] = math.log(length/wordcnt)
#     return wordidf
#
# def tf_idf(wordlist):
#     wordidf = idf(wordlist)
#     tf_idf_list = []
#     for text in wordlist:
#         wordtf = tf(text)
#         word_tf_idf = {}
#         for key in wordtf:
#             word_tf_idf[key] = wordtf[key] * wordidf[key]
#         tf_idf_list.append(word_tf_idf)
#     return tf_idf_list
#
# Using the original methods I write is pretty slow here. I would deploy
# sklearn.
