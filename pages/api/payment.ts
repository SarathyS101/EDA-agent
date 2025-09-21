import { NextApiRequest, NextApiResponse } from 'next'
// PRODUCTION: Uncomment the line below for Stripe integration
// import { createCheckoutSession, verifyWebhookSignature } from '@/lib/stripe'
import { supabaseAdmin } from '@/lib/supabase'
// PRODUCTION: Uncomment the line below for webhook verification
// import { buffer } from 'micro'

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

    // LOCAL TESTING: Skip Stripe and mark as paid immediately
    console.log('🧪 LOCAL TESTING: Skipping Stripe payment, marking as paid')
    
    // Update analysis record directly to paid for local testing
    await supabaseAdmin
      .from('analyses')
      .update({ 
        analysis_status: 'paid',
        payment_intent_id: `local_test_${analysisId}` // Mock payment ID for testing
      })
      .eq('id', analysisId)
      .eq('user_id', userId)

    // Return mock session data for local testing
    res.status(200).json({ 
      sessionId: `local_test_session_${analysisId}`, 
      url: `/dashboard?analysis=${analysisId}&paid=true`,
      localTesting: true 
    })

    /* PRODUCTION: Uncomment the section below for Stripe integration
    
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
    
    */

  } catch (error) {
    console.error('Checkout error:', error)
    res.status(500).json({ message: 'Failed to create checkout session' })
  }
}

