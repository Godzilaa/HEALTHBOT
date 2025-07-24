"use client"

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

export default function CommunityPosts() {
  const [posts, setPosts] = useState([
    { id: 1, author: 'Jane Doe', content: 'Just completed a 5k run! Feeling great!', likes: 5, comments: 2 },
    { id: 2, author: 'John Smith', content: 'Anyone have tips for reducing stress?', likes: 3, comments: 1 },
  ])
  const [newPost, setNewPost] = useState('')
  const [authorName, setAuthorName] = useState('')

  const handlePostSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!newPost.trim() || !authorName.trim()) return
    const newId = posts.length + 1
    setPosts([...posts, { id: newId, author: authorName, content: newPost, likes: 0, comments: 0 }])
    setNewPost('')
    setAuthorName('')
  }

  return (
    <>
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Create a Post</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handlePostSubmit} className="space-y-4">
            <Input
              value={authorName}
              onChange={(e) => setAuthorName(e.target.value)}
              placeholder="Your name"
              required
            />
            <Textarea
              value={newPost}
              onChange={(e) => setNewPost(e.target.value)}
              placeholder="Share your thoughts or ask a question..."
              required
            />
            <Button type="submit">Post</Button>
          </form>
        </CardContent>
      </Card>
      {posts.map((post) => (
        <Card key={post.id} className="mb-4">
          <CardContent className="pt-6">
            <div className="flex items-center mb-4">
              <Avatar className="h-10 w-10 mr-4">
                <AvatarImage src={`https://api.dicebear.com/6.x/initials/svg?seed=${post.author}`} />
                <AvatarFallback>{post.author[0]}</AvatarFallback>
              </Avatar>
              <div>
                <p className="font-semibold">{post.author}</p>
              </div>
            </div>
            <p className="mb-4">{post.content}</p>
            <div className="flex items-center text-sm text-gray-500">
              <Button variant="ghost" size="sm" className="mr-2">Like ({post.likes})</Button>
              <Button variant="ghost" size="sm">Comment ({post.comments})</Button>
            </div>
          </CardContent>
        </Card>
      ))}
    </>
  )
}

