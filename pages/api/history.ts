import { NextApiRequest, NextApiResponse } from 'next'
import { supabaseAdmin } from '@/lib/supabase'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ message: 'Method not allowed' })
  }

  try {
    const { userId } = req.query

    if (!userId) {
      return res.status(400).json({ message: 'Missing user ID' })
    }

    // Get user's analysis history
    const { data: analyses, error } = await supabaseAdmin
      .from('analyses')
      .select(`
        id,
        csv_filename,
        analysis_status,
        pdf_url,
        created_at,
        updated_at,
        analysis_results
      `)
      .eq('user_id', userId)
      .order('created_at', { ascending: false })

    if (error) {
      console.error('History fetch error:', error)
      return res.status(500).json({ message: 'Failed to fetch analysis history' })
    }

    // Get user stats
    const { data: userStats, error: statsError } = await supabaseAdmin
      .from('analyses')
      .select('id', { count: 'exact' })
      .eq('user_id', userId)
      .eq('analysis_status', 'completed')

    const completedAnalyses = userStats?.length || 0

    res.status(200).json({
      analyses: analyses || [],
      stats: {
        totalAnalyses: analyses?.length || 0,
        completedAnalyses,
        pendingAnalyses: (analyses?.length || 0) - completedAnalyses
      }
    })

  } catch (error) {
    console.error('History API error:', error)
    res.status(500).json({ message: 'Internal server error' })
  }
}