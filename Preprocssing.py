from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
from pyvi import ViTokenizer
import nltk
nltk.download('punkt')
from collections import Counter

class Data:
    def __init__(self,path):
        self.path=path
        self.Data=[]
        self.Label=[]
        self.all_word=[]
    def load_data(self):
        dataset=os.listdir(self.path)
        stop_word=[]
        with open('D:\\project1\\stopword.txt',encoding='utf8') as f:
            for line in f:
                stop_word.append(ViTokenizer.tokenize(line).replace(' ','_'))
        index=-1
        for data in dataset:
            index+=1
            datapath=os.listdir(os.path.join(self.path,data))
            for filepath in datapath:
                file_text=open(os.path.join(self.path,data,filepath),encoding='utf16').read().lower()
                file_text=re.sub(r'[0-9-,.():;/%$@!*&^?_#+]',' ',file_text)
                file_text=ViTokenizer.tokenize(file_text)
                file_text=file_text.replace('"','')
                file_text=file_text.replace("'",'')
                file_text = re.sub(r"\s+[a-zA-Z]\s+", " ", file_text)
                words=nltk.word_tokenize(file_text)
                words=[word for word in words if word not in stop_word]
                file_text=' '.join(words)
                for word in words:
                    self.all_word.append(word)
                self.Label.append(index)
                self.Data.append(file_text)
    def process_data(self,all_words):
        features=Counter(all_words)
        most_features=[x for (x,y) in features.most_common() if y>3]
        vectorizer=TfidfVectorizer()
        vocab=vectorizer.fit_transform(most_features)
        X=vectorizer.fit_transform(self.Data)
        return self.Data,self.Label
        