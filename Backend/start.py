#!/usr/bin/env python3
"""
Startup script for AI Health Companion Backend
Optimized for cloud deployment (Render, Heroku, etc.)
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_required_files():
    """Check if all required model files exist"""
    required_files = [
        'chat_model.h5',
        'tokenizer.pickle', 
        'label_encoder.pickle',
        'intents.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        logger.error("Please ensure all model files are uploaded to your deployment")
        return False
    
    logger.info("All required files found ✓")
    return True

def main():
    """Main startup function"""
    logger.info("=== AI Health Companion Backend Startup ===")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if we're in a cloud environment
    port = os.environ.get('PORT')
    if port:
        logger.info(f"Detected cloud environment (PORT={port})")
    else:
        logger.info("Running in local environment")
    
    # Check required files
    if not check_required_files():
        logger.error("Startup failed: Missing required files")
        sys.exit(1)
    
    # Import and start the Flask app
    try:
        logger.info("Importing Flask application...")
        from chat import app, load_model_and_dependencies
        
        # Load model and dependencies
        if not load_model_and_dependencies():
            logger.error("Failed to load model and dependencies")
            sys.exit(1)
        
        # Start the server
        host = '0.0.0.0' if port else '127.0.0.1'
        port = int(port) if port else 5000
        
        logger.info(f"Starting server on {host}:{port}")
        app.run(host=host, port=port, debug=False)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
