from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences


sentences = ["Hello how are you", "I am fine thanks", "what about you" ]  

tokenizer = Tokenizer(num_words = 100, oov_token = '<OOV>')  #tokenizer function used to tokenize each sentence into words and and create a dictionary.

                                                            #num_words = total number of tokenized words oov_token = represent 1 for out of vocabulray words 

tokenizer.fit_on_texts(sentences)                            #tokenize word

word_index = tokenizer.word_index                           #display tokenized word with corresponding index

  
sequences = tokenizer.texts_to_sequences(sentences)          #convert each sentence into sequence


padded = pad_sequences(sequences,maxlen=4)                   #pad all sequence to equal length

print("word_index",word_index)

print("sequences",sequences)

print("padded",padded)

print("----------------------------")

test_data = [
    "what is this",
    "i used to rule the world",
    "mantee"
]

test_seq = tokenizer.texts_to_sequences(test_data)

pad_test = pad_sequences(test_seq,maxlen = 10)

print(pad_test)