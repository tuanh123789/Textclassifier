from sklearn import svm
from Preprocssing import Data
import pickle
from setting import DATA_DIR,ALL_WORD_DIR,MODEL_DIR,MODEL_PATH

if __name__=='__main__':

    train_data=Data(DATA_DIR)
    train_data.load_data()
    all_word=train_data.all_word
    X_train,Label_train=train_data.process_data(all_word)

    classifier=svm.SVC(kernel='linear',C=1,decision_function_shape='ovo')
    classifier.fit(X_train,Label_train)

    pickle.dump(all_word,open(ALL_WORD_DIR,'wb'))
    pickle.dump(classifier,open((MODEL_PATH),'wb'))