
from setting import STOP_WORD_DIR,MODEL_PATH,ALL_WORD_DIR
import pickle
from pyvi import ViTokenizer

def load_model_and_vocab():
    all_word=pickle.load(open(ALL_WORD_DIR,'rb'))
    stop_word=[]
    with open(STOP_WORD_DIR,encoding='utf8') as file:
        for line in file:
            stop_word.append(ViTokenizer.tokenize(line).replace(' ','_'))

    model=pickle.load(open(MODEL_PATH,'rb'))
    return all_word,stop_word,model