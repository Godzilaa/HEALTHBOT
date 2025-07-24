"use client"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { getAIResponse } from "@/lib/ai-model"

export default function Dashboard() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Welcome to Your Dashboard!</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle>Chat with AI Companion</CardTitle>
          </CardHeader>
          <CardContent>
            <AIChat />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Health Stats</CardTitle>
          </CardHeader>
          <CardContent>
            <p>Steps today: 8,423</p>
            <p>Heart rate: 72 bpm</p>
            <p>Sleep last night: 7h 30m</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}


import { useState } from 'react'

function AIChat() {
  const [input, setInput] = useState('')
  const [conversation, setConversation] = useState<{ role: 'user' | 'ai'; content: string }[]>([])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    setConversation(prev => [...prev, { role: 'user', content: input }])

    try {
      const aiResponse = await getAIResponse(input)
      setConversation(prev => [...prev, { role: 'ai', content: aiResponse }])
    } catch (error) {
      console.error("Error getting AI response:", error)
      setConversation(prev => [...prev, { role: 'ai', content: "I'm sorry, I'm having trouble responding right now. Please try again later." }])
    }

    setInput('')
  }

  return (
    <>
      <div className="h-[400px] overflow-y-auto mb-4 p-4 border rounded">
        {conversation.map((msg, index) => (
          <div key={index} className={`mb-4 ${msg.role === 'ai' ? 'text-blue-600 dark:text-blue-400' : 'text-green-600 dark:text-green-400'}`}>
            <strong>{msg.role === 'ai' ? 'AI: ' : 'You: '}</strong>
            {msg.content}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message here..."
        />
        <Button type="submit">Send</Button>
      </form>
    </>
  )
}

