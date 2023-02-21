import numpy as np #handles math
import spacy 
from sklearn.datasets import load_iris #pedals and leaves
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,LSTM, Embedding
from pickle import dump,load
from keras.utils import pad_sequences
from keras.models import load_model


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
train_len = 25 + 1
text_sequences = []
for i in range(train_len, len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

#replaced text to seq of numbers which is an ID for each word
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)
#print(type(tokenizer.index_word))

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
#model=create_model(vocabulary_size + 1,seq_len)

#model.fit(X,y,batch_size=250,epochs=360,verbose=1)

#savemodel
#model.save('chatbot_model.h5')
#dump(tokenizer,open("myTokenizer", 'wb'))

def generate_text(model,tokenizer,seq_len, seed_text, num_gen_words):
    output_text = []
    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([seed_text])[0]
        pad_encoded = pad_sequences([encoded_text],maxlen=seq_len, truncating = 'pre')
        pred_word = model.predict(pad_encoded)
        classes = np.argmax(pred_word,axis=1)
        output_text.append(classes)
        for index ,word in tokenizer.index_word.items():
            if classes == index:
                outWord = word
                break
        seed_text += " "+ outWord
    return seed_text

seed_text = "what is your name"
model = load_model('chatbot_model.h5')
tokenizer = load(open('myTokenizer','rb'))
spoken = generate_text(model,tokenizer,10,seed_text,num_gen_words=25)
print(spoken)

######possible improvements maybe to the tokenizer
###as well as the dataset it needs more training 
####maybe a different model other than LSMT

