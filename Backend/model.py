import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

training_sentences = []
training_labels = []
labels = []
responses = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    responses.append(intent.get('responses', []))
    
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Encode labels
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)

# Tokenize text
vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, truncating='post')

# Build model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.fit(padded_sequences, np.array(training_labels), epochs=200)

# Export model to JSON format
def export_model_to_json(model, tokenizer, label_encoder):
    # Model architecture and weights
    model_json = model.to_json()
    model_weights = model.get_weights()
    
    # Tokenizer configuration
    tokenizer_config = {
        'word_index': tokenizer.word_index,
        'num_words': tokenizer.num_words,
        'oov_token': tokenizer.oov_token
    }
    
    # Label encoder configuration
    label_encoder_config = {
        'classes': label_encoder.classes_.tolist()
    }
    
    # Combine everything into one dictionary
    export_data = {
        'model_config': json.loads(model_json),
        'model_weights': [w.tolist() for w in model_weights],
        'tokenizer': tokenizer_config,
        'label_encoder': label_encoder_config,
        'metadata': {
            'vocab_size': vocab_size,
            'max_len': max_len,
            'embedding_dim': embedding_dim
        }
    }
    
    # Save to JSON file
    with open('chat_model.json', 'w') as f:
        json.dump(export_data, f, indent=2)

# Export the model
export_model_to_json(model, tokenizer, lbl_encoder)

# Additional exports (optional)
model.save('chat_model.h5')  # Traditional HDF5 format
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle)
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(lbl_encoder, f)

print("Model successfully exported in JSON format to 'chat_model.json'")