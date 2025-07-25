import os
import logging

# Suppress TensorFlow warnings and force CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # Force CPU usage, disable GPU

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'            

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

# Force TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

# Import Keras modules with error handling
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    # Fallback for older TensorFlow versions
    from keras.models import load_model
    from keras.preprocessing.sequence import pad_sequences

import pickle
import json
import random

app = Flask(__name__)
CORS(app)

# Global variables
model = None
tokenizer = None
lbl_encoder = None
intents_data = None
max_len = 20

def load_model_and_dependencies():
    """Load model and dependencies with error handling"""
    global model, tokenizer, lbl_encoder, intents_data
    
    try:
        logger.info("Loading AI model and dependencies...")
        
        # Load model
        if os.path.exists('chat_model.h5'):
            model = load_model('chat_model.h5')
            logger.info("Model loaded successfully")
        else:
            logger.error("Model file 'chat_model.h5' not found")
            return False

        # Load tokenizer
        if os.path.exists('tokenizer.pickle'):
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            logger.info("Tokenizer loaded successfully")
        else:
            logger.error("Tokenizer file 'tokenizer.pickle' not found")
            return False

        # Load label encoder
        if os.path.exists('label_encoder.pickle'):
            with open('label_encoder.pickle', 'rb') as f:
                lbl_encoder = pickle.load(f)
            logger.info("Label encoder loaded successfully")
        else:
            logger.error("Label encoder file 'label_encoder.pickle' not found")
            return False

        # Load intents data
        if os.path.exists('intents.json'):
            with open('intents.json', encoding='utf-8') as file:
                intents_data = json.load(file)
            logger.info("Intents data loaded successfully")
        else:
            logger.error("Intents file 'intents.json' not found")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error loading model and dependencies: {str(e)}")
        return False

def get_response_from_intent(predicted_intent):
    """Get a random response for the predicted intent"""
    try:
        for intent in intents_data['intents']:
            if intent['tag'] == predicted_intent:
                responses = intent.get('responses', [])
                if responses:
                    return random.choice(responses)
        return "I'm sorry, I didn't understand that. Could you please rephrase?"
    except Exception as e:
        logger.error(f"Error getting response: {str(e)}")
        return "I'm having trouble generating a response right now."

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'service': 'AI Health Companion Backend',
            'version': '1.0.0'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def predict():
    """Main chat endpoint"""
    try:
        # Check if model is loaded
        if model is None or tokenizer is None or lbl_encoder is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.'
            }), 500
        
        # Get and validate request data
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'error': 'No message provided in request'
            }), 400
            
        message = data['message']
        if not message or not isinstance(message, str):
            return jsonify({
                'error': 'Invalid message format'
            }), 400
        
        message = message.strip()
        if not message:
            return jsonify({
                'error': 'Empty message'
            }), 400
        
        logger.info(f"Processing message: {message[:50]}...")
        
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(sequence, maxlen=max_len, truncating='post')
        
        # Make prediction
        prediction = model.predict(padded, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        
        # Get predicted intent
        predicted_intent = lbl_encoder.classes_[predicted_class]
        
        # Get response
        response_text = get_response_from_intent(predicted_intent)
        
        logger.info(f"Predicted intent: {predicted_intent}, Confidence: {confidence:.3f}")
        
        return jsonify({
            'response': response_text,
            'intent': predicted_intent,
            'confidence': confidence
        }), 200
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Something went wrong processing your request'
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'AI Health Companion Backend API',
        'status': 'running',
        'endpoints': {
            'chat': '/chat (POST)',
            'health': '/health (GET)'
        }
    })

if __name__ == '__main__':
    logger.info("Starting AI Health Companion Backend...")
    
    # Load model and dependencies
    if load_model_and_dependencies():
        logger.info("All dependencies loaded successfully!")
        
        # Get port from environment variable (for Render deployment)
        port = int(os.environ.get('PORT', 5000))
        host = '0.0.0.0' if os.environ.get('PORT') else '127.0.0.1'
        
        logger.info(f"Starting server on {host}:{port}")
        app.run(host=host, port=port, debug=False)
    else:
        logger.error("Failed to load model and dependencies. Exiting.")
        exit(1)