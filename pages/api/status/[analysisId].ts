import { NextApiRequest, NextApiResponse } from 'next'
import { supabaseAdmin } from '@/lib/supabase'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ message: 'Method not allowed' })
  }

  try {
    const { analysisId } = req.query
    const { userId } = req.query

    if (!analysisId || !userId) {
      return res.status(400).json({ message: 'Missing analysisId or userId' })
    }

    // Get analysis status
    const { data: analysis, error } = await supabaseAdmin
      .from('analyses')
      .select('id, analysis_status, pdf_url, updated_at, analysis_results')
      .eq('id', analysisId)
      .eq('user_id', userId)
      .single()

    if (error || !analysis) {
      return res.status(404).json({ message: 'Analysis not found' })
    }

    res.status(200).json({
      analysisId: analysis.id,
      status: analysis.analysis_status,
      pdfUrl: analysis.pdf_url,
      updatedAt: analysis.updated_at,
      hasResults: !!analysis.analysis_results
    })

  } catch (error) {
    console.error('Status check error:', error)
    res.status(500).json({ message: 'Internal server error' })
  }
}