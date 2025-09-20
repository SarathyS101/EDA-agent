import { NextApiRequest, NextApiResponse } from 'next'
import { createCheckoutSession, verifyWebhookSignature } from '@/lib/stripe'
import { supabaseAdmin } from '@/lib/supabase'
import { buffer } from 'micro'

export const config = {
  api: {
    bodyParser: false,
  },
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'POST') {
    return handleCheckout(req, res)
  } else if (req.method === 'POST' && req.url?.includes('webhook')) {
    return handleWebhook(req, res)
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

async function handleWebhook(req: NextApiRequest, res: NextApiResponse) {
  try {
    const buf = await buffer(req)
    const signature = req.headers['stripe-signature'] as string

    const event = verifyWebhookSignature(buf.toString(), signature)

    switch (event.type) {
      case 'checkout.session.completed':
        const session = event.data.object as any
        const userId = session.metadata.userId
        
        // Find and update the analysis record
        const { error } = await supabaseAdmin
          .from('analyses')
          .update({ analysis_status: 'paid' })
          .eq('payment_intent_id', session.payment_intent)
          .eq('user_id', userId)

        if (error) {
          console.error('Failed to update analysis status:', error)
        }

        // Trigger analysis process here
        // You could add the analysis to a queue or call the analyze endpoint
        
        break

      case 'payment_intent.payment_failed':
        const failedPayment = event.data.object as any
        
        await supabaseAdmin
          .from('analyses')
          .update({ analysis_status: 'payment_failed' })
          .eq('payment_intent_id', failedPayment.id)

        break
    }

    res.status(200).json({ received: true })

  } catch (error) {
    console.error('Webhook error:', error)
    res.status(400).json({ message: 'Webhook error' })
  }
}