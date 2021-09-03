from unicodedata import name
import flask
from flask.app import Flask
from flask.json import jsonify
from setting import MODEL_PATH, MODEL_PATH,ALL_WORD_DIR,STOP_WORD_DIR
from flask import request
import pickle
from pyvi import ViTokenizer
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import sys
from flask_cors import CORS
from sklearn import svm
from utils import load_model_and_vocab

app=Flask(__name__,static_url_path='')

all_word,stop_word,model=load_model_and_vocab()

class_name=['Gia dinh','Hackers va Virus','Loi song', 'Khong gian song', 'Duong vao WTO',
            'Cuoc song do day','Am thuc', 'Thoi trang', 'Hinh su', 'Am nhac','San pham tin hoc moi',
            'Tennis','San khau dien anh','Giai tri tin hoc','Kinh doanh quoc te','Cum ga',
            'Du hoc Mua sam','The gioi tre','My thuat','Gioi tinh','Lam dep','Giao duc','Bat dong san',
            'Du lich','Bong da','Chung khoan']

@app.route('/text_classifier/',methods=['POST'])
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
    data=' '.join(words)
    data=[data]

    vectorizer=TfidfVectorizer()
    vocab=vectorizer.fit_transform(all_word)
    test=vectorizer.transform(data)

    class_predict=model.predict(test)

    for index,name in enumerate(class_name):
        if index==class_predict[0]-1:
            output['chủ đề']=name

    return jsonify(output)

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--port', default=8080)
    args = arg_parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)