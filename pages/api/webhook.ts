import { NextApiRequest, NextApiResponse } from 'next'
// PRODUCTION: Uncomment the line below for Stripe webhook verification
// import { verifyWebhookSignature } from '@/lib/stripe'
import { supabaseAdmin } from '@/lib/supabase'
// PRODUCTION: Uncomment the line below for Stripe webhook verification
// import { buffer } from 'micro'

// PRODUCTION: Uncomment the config below for Stripe webhooks
/*
export const config = {
  api: {
    bodyParser: false,
  },
}
*/

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' })
  }

  // LOCAL TESTING: Skip Stripe webhook verification
  console.log('🧪 LOCAL TESTING: Stripe webhooks disabled for local development')
  
  // For local testing, just acknowledge the webhook
  res.status(200).json({ 
    received: true, 
    localTesting: true,
    message: 'Stripe webhooks disabled for local testing' 
  })

  /* PRODUCTION: Uncomment the section below for Stripe webhook handling

  try {
    const buf = await buffer(req)
    const signature = req.headers['stripe-signature'] as string

    if (!signature) {
      return res.status(400).json({ message: 'Missing stripe signature' })
    }

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

  */
}