import numpy as np #handles math
import spacy 
from sklearn.datasets import load_iris #pedals and leaves
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,LSTM, Embedding
from pickle import dump,load


#reading from file
def read_file(file):
    with open(file)as f:
        txt = f.read()
    return txt

#tokenization
nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner', 'lemmatizer'])

def seperate_punch(docText):
    return [token.text.lower() for token in nlp(docText) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n '] #remove all puncuation from string

#call function
d = read_file('newJ.txt')
#tokens
tokens = seperate_punch(d)

#traning to predict next word, can change the length depending onhow long i want the sentence but must include the + 1
train_len = 10 + 1
text_sequences = []
for i in range(train_len, len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

#replaced text to seq of numbers which is an ID for each word
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)


#size of vocab
vocabulary_size = len(tokenizer.word_counts)

#format into a matrix
sequences = np.array(sequences)

#create the LSTM based model
X = sequences[:,:-1] #grabs all rows execpt the last column
y = sequences [:,-1] 

y = to_categorical(y,num_classes=vocabulary_size + 1)
seq_len = X.shape[1] #72141 sequences, 8 words long

#function to create model
def create_model(vocabulary_size,seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size,seq_len,input_length=seq_len)) #defining input dim,output dim, inputlength 
    model.add(LSTM(360, return_sequences=True))
    model.add(LSTM(360))
    model.add(Dense(360,activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    return model
#call the function
model=create_model(vocabulary_size + 1,seq_len)

model.fit(X,y,batch_size=250,epochs=360,verbose=1)

#savemodel
model.save('chatbot_model.h5')
dump(tokenizer,open("myTokenizer", 'wb'))
