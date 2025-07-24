import Link from 'next/link'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

const therapists = [
  { id: 1, name: "Dr. Jane Smith", specialty: "Cognitive Behavioral Therapy", rate: 100, avatar: "JS" },
  { id: 2, name: "Dr. Michael Johnson", specialty: "Family Therapy", rate: 120, avatar: "MJ" },
  { id: 3, name: "Dr. Emily Brown", specialty: "Anxiety and Depression", rate: 90, avatar: "EB" },
]

export default function TherapyPage() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Find a Therapist</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {therapists.map((therapist) => (
          <Card key={therapist.id}>
            <CardHeader>
              <div className="flex items-center space-x-4">
                <Avatar>
                  <AvatarImage src={`https://api.dicebear.com/6.x/initials/svg?seed=${therapist.name}`} />
                  <AvatarFallback>{therapist.avatar}</AvatarFallback>
                </Avatar>
                <div>
                  <CardTitle>{therapist.name}</CardTitle>
                  <p className="text-sm text-gray-500">{therapist.specialty}</p>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <p className="font-semibold">${therapist.rate}/hour</p>
            </CardContent>
            <CardFooter>
              <Button asChild className="w-full">
                <Link href={`/therapy/${therapist.id}`}>View Profile</Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  )
}

