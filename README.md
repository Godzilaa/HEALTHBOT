# HealthBot 🤖💊

An AI-powered health assistant that provides reliable responses to medical queries using a trained intent classification model. Built with **Next.js** (frontend) and **Flask + TensorFlow** (backend), deployed on **Vercel** and **Render**.

[![Frontend](https://img.shields.io/badge/Frontend-Vercel-000000?logo=vercel)](https://healthbot.vercel.app)
[![Backend](https://img.shields.io/badge/Backend-Render-46B3E6?logo=render)](https://healthbot-api.onrender.com)

![HealthBot Demo](demo.gif) <!-- Add a demo GIF if available -->

## Features ✨
- Natural language processing for health-related queries
- ML-powered intent classification (98% accuracy on test data)
- Responsive Next.js frontend with chat interface
- Scalable Flask API deployed on Render

## Tech Stack 🛠️
| Component       | Technologies Used |
|----------------|------------------|
| **Frontend**   | Next.js, TypeScript, TailwindCSS |
| **Backend**    | Flask, TensorFlow/Keras, Python |
| **ML Model**   | Intent Classification (Embedding + Dense Layers) |
| **Deployment** | Vercel (Frontend), Render (Backend) |

## Project Structure 📂
HEALTHBOT/
├── Backend/ # Flask API and ML model
│ ├── app.py # Flask application
│ ├── chat_model.h5 # Trained Keras model
│ ├── intents.json # Training data (patterns/responses)
│ ├── requirements.txt # Python dependencies
│ └── ... # Other model artifacts
├── Frontend/ # Next.js application
│ ├── pages/ # React components
│ ├── public/ # Static assets
│ └── ... # Next.js config files
└── README.md # You are here!


## Setup Instructions 🛠️

### Prerequisites
- Python 3.8+ (Backend)
- Node.js 16+ (Frontend)
- Git

### 1. Backend Setup (Flask API)
```bash
# Clone repository
git clone https://github.com/Godzilaa/HEALTHBOT.git
cd HEALTHBOT/Backend

# Install dependencies
pip install -r requirements.txt

# Run locally (default port 5000)
python app.py

'''

cd ../Frontend

# Install dependencies
npm install

# Run development server (port 3000)
npm run dev
'''
Contributing 🤝
Pull requests welcome! For major changes, please open an issue first.
