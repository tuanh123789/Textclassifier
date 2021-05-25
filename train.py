from sklearn import svm
from Preprocssing import Data
import pickle

train_data=Data('D:\\Textclassifier\\DL_dataset\\new train')
train_data.load_data()
all_word=train_data.all_word
X_train,Label_train=train_data.process_data(all_word)

classifier=svm.SVC(kernel='linear',C=1,decision_function_shape='ovo')
classifier.fit(X_train,Label_train)

pickle.dump(classifier,open('classifier.sav','wb'))