import Link from 'next/link'
import { Button } from "@/components/ui/button"

export default function Home() {
  return (
    <div className="container mx-auto px-4 py-16">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-6">Welcome to AI Health Companion</h1>
        <p className="text-xl mb-8">Your personal AI assistant for health and mental wellness</p>
        <Button asChild>
          <Link href="/dashboard">Get Started</Link>
        </Button>
      </div>
      <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
        <FeatureCard
          title="AI Health Assistant"
          description="Get personalized health advice and answers to your medical questions"
        />
        <FeatureCard
          title="Mental Wellness Companion"
          description="Support for your mental health with guided meditation and mood tracking"
        />
        <FeatureCard
          title="Fall Detection"
          description="Advanced fall detection system to keep you safe"
        />
      </div>
    </div>
  )
}

function FeatureCard({ title, description }: { title: string; description: string }) {
  return (
    <div className="border rounded-lg p-6">
      <h2 className="text-2xl font-semibold mb-4">{title}</h2>
      <p>{description}</p>
    </div>
  )
}

