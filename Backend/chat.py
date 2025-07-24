from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json

app = Flask(__name__)
CORS(app)

# Load model and dependencies
model = load_model('chat_model.h5')  # Your .h5 model file

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as f:
    lbl_encoder = pickle.load(f)

# Load intents for responses
with open('intents.json', encoding='utf-8') as file:
    intents_data = json.load(file)

@app.route('/chat', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data['message']
        
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(sequence, maxlen=20, truncating='post')
        
        # Predict
        prediction = model.predict(padded)
        predicted_label = np.argmax(prediction, axis=1)
        tag = lbl_encoder.inverse_transform(predicted_label)[0]
        
        # Get random response for the predicted tag
        response = get_response(tag)
        
        return jsonify({
            "status": "success",
            "tag": tag,
            "response": response
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

def get_response(tag):
    """Fetch random response from intents.json for the predicted tag"""
    for intent in intents_data['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    return "I didn't understand that. Could you rephrase?"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)