import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_json('Sarcasm_Headlines_Dataset_v2.json', lines=True)


sentence = []
label = []

tokenizer = Tokenizer(num_words = 1000, oov_token = '<OOV>')

for index,data in df.iterrows():
    sentence.append(data['headline'])
    label.append(data['is_sarcastic'])


tokenizer.fit_on_texts(sentence)
word_index = tokenizer.word_index


sequences = tokenizer.texts_to_sequences(sentence)


padded = pad_sequences(sequences,padding = 'post')

print(padded)
