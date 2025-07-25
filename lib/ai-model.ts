import * as tf from '@tensorflow/tfjs';

// Define interfaces for our model data structure
interface ModelData {
  model_config: {
    layers: Array<{
      class_name: string;
      config: {
        input_shape?: number[];
        input_dim?: number;
        output_dim?: number;
        units?: number;
        activation?: 'relu' | 'softmax' | 'sigmoid' | 'tanh' | 'linear';
        name?: string;
      };
    }>;
  };
  weights: number[][][];
  metadata: {
    vocab_size: number;
    max_len: number;
    num_classes: number;
  };
  tokenizer: TokenizerData;
  label_encoder: LabelData;
}

interface TokenizerData {
  word_index: { [key: string]: number };
}

interface LabelData {
  classes_: string[];
}

interface IntentData {
  intents: Array<{
    tag: string;
    responses: string[];
  }>;
}

// Global variables to store loaded model and data
let model: tf.Sequential | null = null;
let tokenizer: TokenizerData | null = null;
let labelEncoder: LabelData | null = null;
let intentsData: IntentData | null = null;
let metadata: ModelData['metadata'] | null = null;

// Load the model and associated data
async function loadModel(): Promise<void> {
  if (model) return; // Already loaded

  try {
    // Load model data from public directory
    const response = await fetch('/Backend/chat_model.json');
    const modelData: ModelData = await response.json();
    
    // Load intents data
    const intentsResponse = await fetch('/Backend/intents.json');
    intentsData = await intentsResponse.json();
    
    // Extract tokenizer and label encoder from model data
    tokenizer = modelData.tokenizer;
    labelEncoder = modelData.label_encoder;
    metadata = modelData.metadata;
    
    // Create TensorFlow.js model manually from the architecture
    model = tf.sequential();
    
    // Add layers based on the exported configuration
    const layers = modelData.model_config.layers;
    
    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      
      if (layer.class_name === 'Embedding') {
        model.add(tf.layers.embedding({
          inputDim: layer.config.input_dim!,
          outputDim: layer.config.output_dim!,
          inputLength: metadata!.max_len,
          name: layer.config.name
        }));
      } else if (layer.class_name === 'GlobalAveragePooling1D') {
        model.add(tf.layers.globalAveragePooling1d({
          name: layer.config.name
        }));
      } else if (layer.class_name === 'Dense') {
        model.add(tf.layers.dense({
          units: layer.config.units!,
          activation: layer.config.activation,
          name: layer.config.name
        }));
      }
    }
    
    // Set the weights
    const weightArrays = modelData.weights.map(layerWeights => 
      layerWeights.map(weightMatrix => tf.tensor(weightMatrix))
    );
    
    model.setWeights(weightArrays.flat());
    
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    throw error;
  }
}

// Preprocess text input
function preprocessText(text: string): tf.Tensor {
  if (!tokenizer || !metadata) {
    throw new Error('Tokenizer or metadata not loaded');
  }
  
  // Convert text to lowercase and split into words
  const words = text.toLowerCase().split(' ');
  
  // Convert words to indices using the tokenizer's word_index
  const sequence = words.map(word => tokenizer!.word_index[word] || 0);
  
  // Pad sequence to max_len
  const paddedSequence = Array(metadata.max_len).fill(0);
  for (let i = 0; i < Math.min(sequence.length, metadata.max_len); i++) {
    paddedSequence[i] = sequence[i];
  }
  
  return tf.tensor2d([paddedSequence]);
}

// Make prediction and get response
async function predict(text: string): Promise<string> {
  if (!model || !labelEncoder || !intentsData || !metadata) {
    throw new Error('Model not loaded');
  }
  
  try {
    // Preprocess input
    const inputTensor = preprocessText(text);
    
    // Make prediction
    const prediction = model.predict(inputTensor) as tf.Tensor;
    const predictionData = await prediction.data();
    
    // Get the predicted class index
    const maxIndex = predictionData.indexOf(Math.max(...predictionData));
    
    // Get the predicted label
    const predictedLabel = labelEncoder.classes_[maxIndex];
    
    // Find the corresponding intent and get a random response
    const intent = intentsData.intents.find(intent => intent.tag === predictedLabel);
    if (intent && intent.responses.length > 0) {
      const randomIndex = Math.floor(Math.random() * intent.responses.length);
      return intent.responses[randomIndex];
    }
    
    return "I'm sorry, I didn't understand that. Could you please rephrase?";
  } catch (error) {
    console.error('Error making prediction:', error);
    return "I'm sorry, I'm having trouble processing your request right now.";
  }
}

// Main function to get AI response
export async function getAIResponse(input: string): Promise<string> {
  try {
    // Load model if not already loaded
    await loadModel();
    
    // Get prediction
    const response = await predict(input);
    return response;
  } catch (error) {
    console.error("Error getting AI response:", error);
    return "I'm sorry, I'm having trouble processing your request right now. Please try again later.";
  }
}
