import type { Metadata, Viewport } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import './globals.css'
import { SidebarNav } from '@/components/sidebar-nav'

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] })
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'FL Monitor — DHSV Fault Detection',
  description: 'Federated learning dashboard for oil well valve fault detection',
}

export const viewport: Viewport = {
  colorScheme: 'light',
  themeColor: 'white',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} bg-background`}>
      <body className="font-sans antialiased">
        <div className="flex h-screen overflow-hidden">
          <SidebarNav />
          <main className="flex-1 overflow-y-auto bg-background" id="main-content">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
