import { NextApiRequest, NextApiResponse } from 'next'

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ message: 'Method not allowed' })
  }

  const checks = {
    server: 'running',
    timestamp: new Date().toISOString(),
    environment: {
      node_version: process.version,
      env_loaded: !!process.env.NODE_ENV,
      has_supabase_url: !!process.env.NEXT_PUBLIC_SUPABASE_URL,
      has_openai_key: !!process.env.OPENAI_API_KEY,
      has_stripe_key: !!process.env.STRIPE_SECRET_KEY,
    },
    paths: {
      upload_endpoint: '/api/upload',
      analyze_endpoint: '/api/analyze', 
      payment_endpoint: '/api/payment',
      history_endpoint: '/api/history'
    }
  }

  res.status(200).json(checks)
}