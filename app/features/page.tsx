import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"

export default function Features() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Features</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <FeatureCard
          title="Fall Detection"
          description="Our advanced fall detection system uses machine learning algorithms to detect falls and automatically alert emergency contacts or services if needed."
        />
        <FeatureCard
          title="Mental Health Support"
          description="Access guided meditation sessions, mood tracking tools, and personalized mental wellness advice from our AI companion."
        />
        <FeatureCard
          title="Health Monitoring"
          description="Track your vital signs, sleep patterns, and physical activity to get a comprehensive view of your health status."
        />
        <FeatureCard
          title="Medication Reminders"
          description="Never miss a dose with our smart medication reminder system, which can also warn about potential drug interactions."
        />
      </div>
    </div>
  )
}

function FeatureCard({ title, description }: { title: string; description: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <p>{description}</p>
      </CardContent>
    </Card>
  )
}

