import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import { loadStripe } from '@stripe/stripe-js'
import { CreditCard, ArrowLeft, Shield, Zap } from 'lucide-react'
import axios from 'axios'
import { useAuth } from '@/contexts/AuthContext'
import { ProtectedRoute } from '@/components/ProtectedRoute'
import toast from 'react-hot-toast'

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!)

function PaymentContent() {
  const [loading, setLoading] = useState(false)
  const [analysisId, setAnalysisId] = useState<string | null>(null)
  const { user } = useAuth()
  const router = useRouter()

  useEffect(() => {
    const { analysisId: queryAnalysisId } = router.query
    if (queryAnalysisId && typeof queryAnalysisId === 'string') {
      setAnalysisId(queryAnalysisId)
    }
  }, [router.query])

  const handlePayment = async () => {
    if (!user || !analysisId) return

    setLoading(true)
    
    try {
      const response = await axios.post('/api/payment', {
        userId: user.id,
        analysisId: analysisId
      })

      if (response.data.url) {
        window.location.href = response.data.url
      }
    } catch (error) {
      console.error('Payment setup failed:', error)
      toast.error('Payment setup failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  if (!user) {
    return <div className="min-h-screen flex items-center justify-center">Loading...</div>
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="max-w-2xl mx-auto px-4 py-16">
        <div className="text-center mb-8">
          <button
            onClick={() => router.back()}
            className="inline-flex items-center text-blue-600 hover:text-blue-800 mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </button>
          
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            Complete Your Purchase
          </h1>
          <p className="text-gray-600">
            Get your comprehensive EDA report powered by agentic AI
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8">
          {/* Order Summary */}
          <div className="border-b border-gray-200 pb-6 mb-6">
            <h2 className="text-lg font-semibold mb-4">Order Summary</h2>
            <div className="flex justify-between items-center">
              <div>
                <p className="font-medium text-gray-900">AI-Powered EDA Analysis</p>
                <p className="text-sm text-gray-600">Professional PDF report with insights</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-blue-600">$1.00</p>
                <p className="text-sm text-gray-500">One-time payment</p>
              </div>
            </div>
          </div>

          {/* What's Included */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold mb-4">What's Included</h3>
            <div className="space-y-3">
              {[
                'Complete statistical analysis of your dataset',
                'AI-validated insights and recommendations', 
                'Professional visualizations and charts',
                'Data quality assessment and outlier detection',
                'Correlation analysis and distribution plots',
                'Downloadable PDF report (typically 10-20 pages)',
                'Secure processing and data privacy'
              ].map((feature, index) => (
                <div key={index} className="flex items-center">
                  <Zap className="h-4 w-4 text-green-500 mr-3 flex-shrink-0" />
                  <span className="text-sm text-gray-700">{feature}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Security Notice */}
          <div className="bg-blue-50 rounded-lg p-4 mb-6">
            <div className="flex items-start">
              <Shield className="h-5 w-5 text-blue-600 mt-0.5 mr-3" />
              <div className="text-sm">
                <p className="font-medium text-blue-800 mb-1">Secure Payment</p>
                <p className="text-blue-700">
                  Payments are processed securely through Stripe. Your payment information 
                  is encrypted and never stored on our servers.
                </p>
              </div>
            </div>
          </div>

          {/* Payment Button */}
          <button
            onClick={handlePayment}
            disabled={loading || !analysisId}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-4 px-6 rounded-lg transition-colors flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                Setting up payment...
              </>
            ) : (
              <>
                <CreditCard className="h-5 w-5 mr-3" />
                Pay $1.00 - Proceed to Checkout
              </>
            )}
          </button>

          <p className="text-xs text-gray-500 text-center mt-4">
            By proceeding, you agree to our Terms of Service and Privacy Policy. 
            Analysis typically completes within 2-5 minutes after payment.
          </p>
        </div>

        {/* FAQ */}
        <div className="mt-12 bg-white rounded-lg p-6">
          <h3 className="font-semibold mb-4">Frequently Asked Questions</h3>
          <div className="space-y-4 text-sm">
            <div>
              <p className="font-medium text-gray-900">How long does analysis take?</p>
              <p className="text-gray-600">Most analyses complete within 2-5 minutes after payment confirmation.</p>
            </div>
            <div>
              <p className="font-medium text-gray-900">What file formats are supported?</p>
              <p className="text-gray-600">Currently, we support CSV files up to 10MB in size.</p>
            </div>
            <div>
              <p className="font-medium text-gray-900">Is my data secure?</p>
              <p className="text-gray-600">Yes, your data is processed securely and automatically deleted after analysis completion.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Payment() {
  return (
    <ProtectedRoute>
      <PaymentContent />
    </ProtectedRoute>
  )
}