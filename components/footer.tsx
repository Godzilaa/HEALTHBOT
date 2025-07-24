import Link from 'next/link'

export default function Footer() {
  return (
    <footer className="border-t">
      <div className="container flex items-center justify-between py-4">
        <p>&copy; 2023 AI Health Companion. All rights reserved.</p>
        <nav className="flex space-x-4">
          <Link href="/privacy">Privacy Policy</Link>
          <Link href="/terms">Terms of Service</Link>
          <Link href="/contact">Contact Us</Link>
        </nav>
      </div>
    </footer>
  )
}

