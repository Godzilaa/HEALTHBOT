# HealthBot 🤖💊

![HealthBot Logo](https://via.placeholder.com/150x50?text=HealthBot)  
*AI-powered health assistant for reliable medical information*

## Table of Contents
- [Features](#features-)
- [Tech Stack](#tech-stack-)
- [Installation](#installation-)
- [Usage](#usage-)
- [API Documentation](#api-documentation-)
- [Datasets](#datasets-)
- [Deployment](#deployment-)
- [Contributing](#contributing-)
- [License](#license-)

## Features ✨
- **Symptom Analysis**: Get preliminary assessments of symptoms
- **Medication Info**: Learn about medications and side effects
- **Mental Health Support**: Specialized mental health Q&A
- **Multi-language**: Supports English and Spanish
- **Privacy Focused**: No personal data storage

## Tech Stack 🛠️
| Component       | Technology |
|-----------------|------------|
| Frontend        | Next.js 13, TypeScript, TailwindCSS |
| Backend         | Python 3.9, Flask 2.3, TensorFlow 2.12 |
| NLP Model       | Custom Intent Classification (98% accuracy) |
| Deployment      | Vercel (Frontend), Render (Backend) |

## Installation ⚙️
### 1. Clone Repository
```bash
git clone https://github.com/yourusername/healthbot.git
cd healthbot
```
###2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```
###3. Frontend Setup
```bash
cd ../frontend
npm install
```
##Usage 🚀
###Running Locally
```bash
# In backend directory
flask run --port=5000

# In frontend directory (new terminal)
npm run dev
```

##Datasets
https://www.kaggle.com/code/jocelyndumlao/chatbot-for-mental-health-conversations/input

###
Contributing 🤝
Fork the repository

Create your feature branch (git checkout -b feature/your-feature)

Commit your changes (git commit -m 'Add some feature')

Push to the branch (git push origin feature/your-feature)

Open a Pull Request


