import flask
from flask.app import Flask
from flask.json import jsonify
from setting import MODEL_DIR, MODEL_PATH,ALL_WORD_DIR,STOP_WORD_DIR
from flask import request
import pickle
from pyvi import ViTokenizer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import sys

app=Flask(__name__,static_url_path='')

all_word=pickle.load(open(MODEL_PATH,'rb'))
stop_word=[]
with open(STOP_WORD_DIR,encoding='utf8') as file:
    for line in file:
        stop_word.append(ViTokenizer.tokenize(line).replace(' ','_'))
model=pickle.load(open(MODEL_DIR,'rb'))

class_name=['Am nhac','Am thuc','Bat dong san','Bong da','Chung khoan','Cum ga','Cuoc song do day','Du hoc',
           'Du lich','Duong vao WTO','Gia dinh','Giai tri tin hoc','Giao duc','Gioi tinh','Hackers va Virus','Hinh su',
           'Khong gian song','Kinh doanh quoc te','Lam dep','Loi song','Mua sam','My thuat','San khau dien anh','San pham tin hoc moi',
           'Tennis','The gioi tre','Thoi trang']

@app.route('text_classifier',method=['POST'])
def text_classifier():
    output={}
    raw_data=request.form
    data=raw_data['text']
    data=re.sub(r'[0-9-,.():;/%$@!*&^?_#+]',' ',data)
    data=ViTokenizer.tokenize(data)
    data=data.replace('"','')
    data=data.replace("'",'')
    paragraph = re.sub(r"\s+[a-zA-Z]\s+", " ",data)
    words=nltk.word_tokenize(data)
    words=[word for word in words if word not in stop_word]
    paragraph=' '.join(words)

    vectorizer=TfidfVectorizer()
    vocab=vectorizer.fit_transform(all_word)
    test=vectorizer.transform(paragraph)

    class_predict=model.predict(test)

    for index,class in enumerate(class_name):
        if index==class_predict:
            output['chủ đề']=class

    return jsonify(output)

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--port', default=8080)
    args = arg_parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)