# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:20:19 2018
Generate user and item vectors by learning their profiles, item details, and reviews
@author: Feng Zhu
Function Doc2vec for Cross-Domain Recommendation (Datasets: DoubanMoive & DoubanBook)
Based on the model Dov2vec:"https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec"
"""
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from stanfordcorenlp import StanfordCoreNLP
import pymysql.cursors
user_size=2718 #total users in 3 domains
movie_size=9565
book_size=6777
music_size=5567
vector_nums=[8,16,32,64,128]
def is_chinese(uchar):
    """is this a chinese word?"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def is_number(uchar):
    """is this unicode a number?"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False

def is_alphabet(uchar):
    """is this unicode an English word?"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False
    
def format_str(content,lag):
    content_str = ''
    if lag==0: #English
       for i in content:
           if is_alphabet(i):
               content_str = content_str+i
    if lag==1: #Chinese
        for i in content:
            if is_chinese(i):
                content_str = content_str+i
    if lag==2: #Number
        for i in content:
            if is_number(i):
                content_str = content_str+i        
    return content_str
    
#get chinese nlp
nlp = StanfordCoreNLP('stanford-corenlp-full-2018-01-31/',lang='zh')
#clean stops
nltk.download('stopwords')
# Connect to mysql 
# Configure a connector
connect = pymysql.Connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='123456',
    db='douban-reviews',
    charset='utf8'
)

################################Movie Domain########################################################
#All documents (for items and users respectively)
documents=[]
# For users
# User profiles
# For users (put user profiles and reviews together as a user document)
cursor1=connect.cursor()
sql ="SELECT * FROM users_cleaned order by UID"
cursor1.execute(sql)
for row in cursor1.fetchall():
    #row[1]:living place, row[3]:self statement
    str_cleaned=''
    if row[1] is not None:
        str_cleaned=format_str(row[1],1)
    if row[3] is not None:
        str_cleaned=str_cleaned+format_str(row[3],1)
    if str_cleaned=='':
       documents.append([])
       continue
    words= nlp.word_tokenize(str_cleaned)
    documents.append(words)

cursor1.close()
print("Load User Profiles: Finished!")
user_size=len(documents)
print("User_size:%07d"%(user_size))
# User and item reviews (DoubanBook)
for i in range(movie_size):
    documents.append([]) 
# For items (DoubanMovie, item details, mainly of item summary)
# get a cursor
cursor = connect.cursor()
sql = "SELECT * FROM movie_subset_5 order by UID2" 
cursor.execute(sql)
for row in cursor.fetchall():
    str_cleaned=''
    # row[1]:item_name row[2]:director row[3]:summary  row[4]:writer  row[5]:country row[7]: language row[11]:tags, r[15]:UID2
    if row[1] is not None:
        str_cleaned+=format_str(row[1],1)
        
    if row[2] is not None:
        str_cleaned+=format_str(row[2],1)
        
    if row[3] is not None:
        str_cleaned+=format_str(row[3],1)
        
    if row[4] is not None:
        str_cleaned+=format_str(row[4],1)
        
    if row[5] is not None:
        str_cleaned+=format_str(row[5],1)
    
    if row[7] is not None:
        str_cleaned+=format_str(row[7],1)
        
    if row[11] is not None:
        str_cleaned+=format_str(row[11],1)
        
    if str_cleaned=='':
        continue
    words= nlp.word_tokenize(str_cleaned)
    documents[row[15]-1]=documents[row[15]-1]+words
print("Movie_size:%07d"%(movie_size))
cursor.close()
print("Load Moive Details (DoubanMovie): Finished!")

# User reviews (DoubanMovie)
cursor2=connect.cursor()
sql ="SELECT * FROM moviereviews_subset_5 order by user_id"
cursor2.execute(sql)
for row in cursor2.fetchall():
    str_cleaned=''
    if row[3] is not None:
        str_cleaned+=format_str(row[3],1)
    if row[5] is not None:
        str_cleaned+=format_str(row[5],1)
    if str_cleaned=='':
        continue
    words= nlp.word_tokenize(str_cleaned)
    if row[0]<=len(documents):
        documents[row[0]-1]=documents[row[0]-1]+words
    if (user_size+row[1]-1)<len(documents):
        documents[user_size+row[1]-1]=documents[user_size+row[1]-1]+words
cursor2.close()
print("Load User Reviews (DoubanMoive): Finished!")

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in documents:
     for token in text:
         frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
          for text in documents]

# train the model
documents = [TaggedDocument(doc, [int(i)]) for i, doc in enumerate(documents)]
# documents: Users + Movies
for i in vector_nums:
    docs=documents
    model = Doc2Vec(docs, vector_size=i, window=2, min_count=5,negative=5, workers=6)
    model.train(docs,total_examples=model.corpus_count, epochs=50)
    model.save("Doc2vec_douban_movie_VSize%02d.model"%(i))

################################Book Domain########################################################
#All documents (for items and users respectively)
documents=[]
# For users
# User profiles
# For users (put user profiles and reviews together as a user document)
cursor1=connect.cursor()
sql ="SELECT * FROM users_cleaned order by UID"
cursor1.execute(sql)
for row in cursor1.fetchall():
    #row[1]:living place, row[3]:self statement
    str_cleaned=''
    if row[1] is not None:
        str_cleaned=format_str(row[1],1)
    if row[3] is not None:
        str_cleaned=str_cleaned+format_str(row[3],1)
    if str_cleaned=='':
       documents.append([])
       continue
    words= nlp.word_tokenize(str_cleaned)
    documents.append(words)

cursor1.close()
print("Load User Profiles: Finished!")
# User and item reviews (DoubanBook)
for i in range(book_size):
    documents.append([]) 
cursor3=connect.cursor()
sql ="SELECT * FROM bookreviews_subset_5 order by user_id"
cursor3.execute(sql)
for row in cursor3.fetchall():
    str_cleaned=''
    if row[3] is not None:
        str_cleaned+=format_str(row[3],1)
    if row[4] is not None:
        str_cleaned+=format_str(row[4],1)
    if str_cleaned=='':
        continue
    words= nlp.word_tokenize(str_cleaned)
    if row[0]<=len(documents):
        documents[row[0]-1]=documents[row[0]-1]+words
        documents[user_size+row[1]-1]=documents[user_size+row[1]-1]+words
cursor3.close()
print("Load User Reviews: Finished!")

# train the model
documents = [TaggedDocument(doc, [int(i)]) for i, doc in enumerate(documents)]
# documents: Users+Books
for i in vector_nums:
    docs=documents
    model = Doc2Vec(docs, vector_size=i, window=2, min_count=5,negative=5, workers=6)
    model.train(docs,total_examples=model.corpus_count, epochs=50)
    model.save("Doc2vec_douban_book_VSize%02d.model"%(i))        


################################Music Domain########################################################
#All documents (for items and users respectively)
documents=[]
# For users
# User profiles
# For users (put user profiles and reviews together as a user document)
cursor1=connect.cursor()
sql ="SELECT * FROM users_cleaned order by UID"
cursor1.execute(sql)
for row in cursor1.fetchall():
    str_cleaned=''
    if row[1] is not None:
        str_cleaned=format_str(row[1],1)
    if row[3] is not None:
        str_cleaned=str_cleaned+format_str(row[3],1)
    if str_cleaned=='':
       documents.append([])
       continue
    words= nlp.word_tokenize(str_cleaned)
    documents.append(words)

cursor1.close()
print("Load User Profiles: Finished!")
# User and item reviews (DoubanMusic)
for i in range(music_size):
    documents.append([]) 
cursor3=connect.cursor()
sql ="SELECT * FROM musicreviews_subset_5 order by user_id"
cursor3.execute(sql)
for row in cursor3.fetchall():
    str_cleaned=''
    if row[3] is not None:
        str_cleaned+=format_str(row[3],1)
    if row[4] is not None:
        str_cleaned+=format_str(row[4],1)
    if str_cleaned=='':
        continue
    words= nlp.word_tokenize(str_cleaned)
    if row[0]<=len(documents):
        documents[row[0]-1]=documents[row[0]-1]+words
        documents[user_size+row[1]-1]=documents[user_size+row[1]-1]+words
cursor3.close()
print("Load User and Music Reviews: Finished!")

# train the model
documents = [TaggedDocument(doc, [int(i)]) for i, doc in enumerate(documents)]
# documents: Users+Music
for i in vector_nums:
    docs=documents
    model = Doc2Vec(docs, vector_size=i, window=2, min_count=5,negative=5, workers=6)
    model.train(docs,total_examples=model.corpus_count, epochs=50)
    model.save("Doc2vec_douban_music_VSize%02d.model"%(i)) 
    
