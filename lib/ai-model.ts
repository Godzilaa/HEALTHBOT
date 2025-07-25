// AI Model API Client
// This module handles communication with the AI chatbot backend

interface ChatResponse {
  response: string;
  confidence?: number;
  intent?: string;
}

interface ErrorResponse {
  error: string;
  message?: string;
}

// Configuration for the AI backend
const AI_CONFIG = {
  // Use environment variable or fallback to localhost
  baseUrl: process.env.NEXT_PUBLIC_AI_API_URL || 'http://127.0.0.1:5000',
  endpoints: {
    chat: '/chat',
    health: '/health'
  },
  timeout: 10000, // 10 second timeout
  retries: 3
};

// Utility function to delay between retries
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

// Main function to get AI response via API call
export async function getAIResponse(input: string): Promise<string> {
  // Input validation
  if (!input || typeof input !== 'string' || input.trim().length === 0) {
    return "Please enter a message for me to respond to.";
  }

  // Sanitize input (basic protection)
  const sanitizedInput = input.trim().slice(0, 1000); // Limit length

  let lastError: Error | null = null;

  // Retry logic for better reliability
  for (let attempt = 1; attempt <= AI_CONFIG.retries; attempt++) {
    try {
      console.log(`AI API attempt ${attempt}/${AI_CONFIG.retries}`);
      
      // Create abort controller for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), AI_CONFIG.timeout);

      const response = await fetch(`${AI_CONFIG.baseUrl}${AI_CONFIG.endpoints.chat}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ 
          message: sanitizedInput,
          timestamp: new Date().toISOString()
        }),
        signal: controller.signal
      });

      // Clear timeout
      clearTimeout(timeoutId);

      // Check if response is ok
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      // Parse response
      const data = await response.json() as ChatResponse | ErrorResponse;

      // Check for error in response data
      if ('error' in data) {
        throw new Error(data.error || 'Unknown server error');
      }

      // Extract and validate response
      const aiResponse = data.response;
      if (!aiResponse || typeof aiResponse !== 'string') {
        throw new Error('Invalid response format from AI service');
      }

      console.log('AI response received successfully');
      return aiResponse;

    } catch (error) {
      lastError = error as Error;
      console.error(`AI API attempt ${attempt} failed:`, error);

      // Don't retry on certain errors
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.error('Request timed out');
        } else if (error.message.includes('HTTP 4')) {
          // Don't retry on 4xx errors (client errors)
          break;
        }
      }

      // Wait before retrying (exponential backoff)
      if (attempt < AI_CONFIG.retries) {
        const waitTime = Math.min(1000 * Math.pow(2, attempt - 1), 5000);
        console.log(`Waiting ${waitTime}ms before retry...`);
        await delay(waitTime);
      }
    }
  }

  // All retries failed
  console.error('All AI API attempts failed:', lastError);
  
  // Return user-friendly error message based on the type of error
  if (lastError?.name === 'AbortError') {
    return "The AI service is taking too long to respond. Please try again.";
  } else if (lastError?.message.includes('Failed to fetch') || lastError?.message.includes('NetworkError')) {
    return "I'm unable to connect to the AI service right now. Please check your internet connection and try again.";
  } else if (lastError?.message.includes('HTTP 5')) {
    return "The AI service is temporarily unavailable. Please try again in a few moments.";
  } else {
    return "I'm sorry, I'm having trouble processing your request right now. Please try again later.";
  }
}

// Health check function to test if the AI service is available
export async function checkAIServiceHealth(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout for health check

    const response = await fetch(`${AI_CONFIG.baseUrl}${AI_CONFIG.endpoints.health}`, {
      method: 'GET',
      signal: controller.signal
    });

    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    console.error('AI service health check failed:', error);
    return false;
  }
}

// Function to get AI service status for debugging
export async function getAIServiceStatus(): Promise<{ 
  available: boolean; 
  baseUrl: string; 
  lastChecked: string;
}> {
  const isAvailable = await checkAIServiceHealth();
  
  return {
    available: isAvailable,
    baseUrl: AI_CONFIG.baseUrl,
    lastChecked: new Date().toISOString()
  };
}
