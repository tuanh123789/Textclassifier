import pickle
import re
from pyvi import ViTokenizer
import nltk
nltk.download('punkt')
from Train import all_word
from sklearn.feature_extraction.text import TfidfVectorizer

classifier=pickle.load(open('classifier.sav','rb'))
print('Nhap vao noi dung van ban can xu li:')
paragraph=input()

class_name=['Am nhac','Am thuc','Bat dong san','Bong da','Chung khoan','Cum ga','Cuoc song do day','Du hoc',
           'Du lich','Duong vao WTO','Gia dinh','Giai tri tin hoc','Giao duc','Gioi tinh','Hackers va Virus','Hinh su',
           'Khong gian song','Kinh doanh quoc te','Lam dep','Loi song','Mua sam','My thuat','San khau dien anh','San pham tin hoc moi',
           'Tennis','The gioi tre','Thoi trang']

stop_word=[]
with open('D:\\project1\\stopword.txt',encoding='utf8') as f:
    for line in f:
        stop_word.append(ViTokenizer.tokenize(line).replace(' ','_'))

paragraph=re.sub(r'[0-9-,.():;/%$@!*&^?_#+]',' ',paragraph)
paragraph=ViTokenizer.tokenize(paragraph)
paragraph=file_text.replace('"','')
paragraph=file_text.replace("'",'')
paragraph = re.sub(r"\s+[a-zA-Z]\s+", " ",paragraph)
words=nltk.word_tokenize(paragraph)
words=[word for word in words if word not in stop_word]
paragraph=' '.join(words)

vectorizer=TfidfVectorizer()
vocab=vectorizer.fit_transform(all_word)
test=vectorizer.transform(paragraph)

class_predict=classifier.predict(test)

for i,j in enumerate(class_name):
    if i==class_predict:
        print('Van ban thuoc vao chu de: {}'.format(j))
