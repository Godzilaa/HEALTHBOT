export async function processPayment(amount: number): Promise<{ success: boolean, transactionId: string }> {
  // This is a mock function. In a real application, you would integrate with a payment gateway.
  return new Promise((resolve) => {
    setTimeout(() => {
      const success = Math.random() > 0.1; // 90% success rate
      resolve({
        success,
        transactionId: success ? `TXN${Math.random().toString(36).substr(2, 9)}` : '',
      });
    }, 2000); // Simulate network delay
  });
}

