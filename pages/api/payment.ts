import { NextApiRequest, NextApiResponse } from 'next'
import { createCheckoutSession, verifyWebhookSignature } from '@/lib/stripe'
import { supabaseAdmin } from '@/lib/supabase'
import { buffer } from 'micro'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'POST') {
    return handleCheckout(req, res)
  } else {
    return res.status(405).json({ message: 'Method not allowed' })
  }
}

async function handleCheckout(req: NextApiRequest, res: NextApiResponse) {
  try {
    const { userId, analysisId } = req.body

    if (!userId || !analysisId) {
      return res.status(400).json({ message: 'Missing required fields' })
    }

    // Create Stripe checkout session
    const session = await createCheckoutSession('price_eda_analysis', userId)

    // Update analysis record with payment intent
    await supabaseAdmin
      .from('analyses')
      .update({ 
        payment_intent_id: session.payment_intent as string,
        analysis_status: 'pending_payment' 
      })
      .eq('id', analysisId)
      .eq('user_id', userId)

    res.status(200).json({ sessionId: session.id, url: session.url })

  } catch (error) {
    console.error('Checkout error:', error)
    res.status(500).json({ message: 'Failed to create checkout session' })
  }
}

