import os
DIR_PATH_DATA = os.path.dirname(os.path.realpath(__file__)) + "/data/"
MODEL_DIR=os.path.dirname(os.path.realpath(__file__)) + "/model/"

MODEL_PATH=MODEL_DIR+'classifier.sav'

DATA_DIR=DIR_PATH_DATA+'new_train'
TEST_DIR=DIR_PATH_DATA+'test'
STOP_WORD_DIR=DIR_PATH_DATA+'stopword.txt'
ALL_WORD_DIR=DIR_PATH_DATA+'all_word.txt'