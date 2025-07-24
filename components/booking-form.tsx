"use client"

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { toast } from "@/components/ui/use-toast"
import { processPayment } from "@/lib/payment-processing"

export default function BookingForm({ therapistId, therapistName, rate }: { therapistId: number, therapistName: string, rate: number }) {
  const [date, setDate] = useState('')
  const [time, setTime] = useState('')
  const [duration, setDuration] = useState('1')
  const [isProcessing, setIsProcessing] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsProcessing(true)

    const totalAmount = rate * parseFloat(duration)

    try {
      const { success, transactionId } = await processPayment(totalAmount)

      if (success) {
        toast({
          title: "Booking Confirmed",
          description: `Your session with ${therapistName} has been booked for ${date} at ${time}. Transaction ID: ${transactionId}`,
        })
        // Here you would typically send the booking data to your backend
        console.log('Booking submitted:', { therapistId, date, time, duration, transactionId })
      } else {
        toast({
          title: "Payment Failed",
          description: "There was an error processing your payment. Please try again.",
          variant: "destructive",
        })
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "An unexpected error occurred. Please try again later.",
        variant: "destructive",
      })
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <Label htmlFor="date">Date</Label>
        <Input
          id="date"
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          required
        />
      </div>
      <div>
        <Label htmlFor="time">Time</Label>
        <Input
          id="time"
          type="time"
          value={time}
          onChange={(e) => setTime(e.target.value)}
          required
        />
      </div>
      <div>
        <Label htmlFor="duration">Duration</Label>
        <Select value={duration} onValueChange={setDuration}>
          <SelectTrigger>
            <SelectValue placeholder="Select duration" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1">1 hour</SelectItem>
            <SelectItem value="1.5">1.5 hours</SelectItem>
            <SelectItem value="2">2 hours</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <p className="font-semibold">Total: ${rate * parseFloat(duration)}</p>
      <Button type="submit" className="w-full" disabled={isProcessing}>
        {isProcessing ? "Processing..." : "Book and Pay"}
      </Button>
    </form>
  )
}

