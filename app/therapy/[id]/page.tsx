import { notFound } from 'next/navigation'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import BookingForm from '@/components/booking-form'

const therapists = [
  { id: 1, name: "Dr. Jane Smith", specialty: "Cognitive Behavioral Therapy", rate: 100, avatar: "JS", bio: "Dr. Jane Smith is a licensed psychologist with over 10 years of experience in Cognitive Behavioral Therapy. She specializes in treating anxiety, depression, and stress-related disorders." },
  { id: 2, name: "Dr. Michael Johnson", specialty: "Family Therapy", rate: 120, avatar: "MJ", bio: "Dr. Michael Johnson is a family therapist with extensive experience in helping families improve communication and resolve conflicts. He also works with couples on relationship issues." },
  { id: 3, name: "Dr. Emily Brown", specialty: "Anxiety and Depression", rate: 90, avatar: "EB", bio: "Dr. Emily Brown is a clinical psychologist specializing in the treatment of anxiety and depression. She uses a combination of cognitive-behavioral and mindfulness-based approaches in her practice." },
]

export default function TherapistProfile({ params }: { params: { id: string } }) {
  const therapist = therapists.find(t => t.id === parseInt(params.id))

  if (!therapist) {
    notFound()
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-4">
            <Avatar className="h-20 w-20">
              <AvatarImage src={`https://api.dicebear.com/6.x/initials/svg?seed=${therapist.name}`} />
              <AvatarFallback>{therapist.avatar}</AvatarFallback>
            </Avatar>
            <div>
              <CardTitle className="text-2xl">{therapist.name}</CardTitle>
              <p className="text-gray-500">{therapist.specialty}</p>
              <p className="font-semibold">${therapist.rate}/hour</p>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <h2 className="text-xl font-semibold mb-2">About</h2>
          <p className="mb-4">{therapist.bio}</p>
          <h2 className="text-xl font-semibold mb-2">Book a Session</h2>
          <BookingForm therapistId={therapist.id} therapistName={therapist.name} rate={therapist.rate} />
        </CardContent>
      </Card>
    </div>
  )
}

