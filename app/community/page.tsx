import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import CommunityPosts from "@/components/community-posts"

export default function Community() {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-6">Community</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2">
          <CommunityPosts />
        </div>
        <Card>
          <CardHeader>
            <CardTitle>Community Stats</CardTitle>
          </CardHeader>
          <CardContent>
            <p>Total Members: 1,234</p>
            <p>Active Discussions: 56</p>
            <p>Posts Today: 89</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

