import '@/styles/globals.css'
import type { AppProps } from 'next/app'
import { createClientComponentClient } from '@supabase/auth-helpers-nextjs'
import { SessionContextProvider } from '@supabase/auth-helpers-react'
import { useState } from 'react'
import { loadStripe } from '@stripe/stripe-js'
import { Elements } from '@stripe/react-stripe-js'
import { AuthProvider } from '@/contexts/AuthContext'
import { Toaster } from 'react-hot-toast'
import ErrorBoundary from '@/components/ErrorBoundary'

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!)

export default function App({ Component, pageProps }: AppProps) {
  const [supabaseClient] = useState(() => createClientComponentClient())

  return (
    <ErrorBoundary>
      <SessionContextProvider
        supabaseClient={supabaseClient}
        initialSession={pageProps.initialSession}
      >
        <AuthProvider>
          <Elements stripe={stripePromise}>
            <div className="min-h-screen bg-gray-50">
              <Component {...pageProps} />
              <Toaster 
                position="top-right"
                toastOptions={{
                  duration: 4000,
                  style: {
                    background: '#363636',
                    color: '#fff',
                  }
                }}
              />
            </div>
          </Elements>
        </AuthProvider>
      </SessionContextProvider>
    </ErrorBoundary>
  )
}