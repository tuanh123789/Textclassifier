import pickle
from Preprocssing import Data
from sklearn.metrics import classification_report

train_data=Data('D:\\Textclassifier\\DL_dataset\\new train')
train_data.load_data()
all_word=train_data.all_word
test_data=Data('D:\\Textclassifier\\DL_dataset\\new test')
test_data.load_data()
X_test,Label_test=test_data.process_data(all_word)

classifier=pickle.load(open('classifier.sav','rb'))

class_name=['Am nhac','Am thuc','Bat dong san','Bong da','Chung khoan','Cum ga','Cuoc song do day','Du hoc',
           'Du lich','Duong vao WTO','Gia dinh','Giai tri tin hoc','Giao duc','Gioi tinh','Hackers va Virus','Hinh su',
           'Khong gian song','Kinh doanh quoc te','Lam dep','Loi song','Mua sam','My thuat','San khau dien anh','San pham tin hoc moi',
           'Tennis','The gioi tre','Thoi trang']

print(classification_report(Label_test, Label_predict, target_names=class_name))
