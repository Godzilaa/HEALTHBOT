import Link from 'next/link'
import { Button } from "@/components/ui/button"
import { ModeToggle } from "@/components/mode-toggle"

export default function Header() {
  return (
    <header className="border-b">
      <div className="container flex items-center justify-between py-4">
        <Link href="/" className="text-2xl font-bold">
          AI Health Companion
        </Link>
        <nav className="flex items-center space-x-4">
          <Link href="/dashboard">Dashboard</Link>
          <Link href="/features">Features</Link>
          <Link href="/community">Community</Link>
          <Link href="/therapy">Therapy</Link>
          <Link href="/about">About</Link>
          <ModeToggle />
          <Button asChild>
            <Link href="/dashboard">Get Started</Link>
          </Button>
        </nav>
      </div>
    </header>
  )
}

