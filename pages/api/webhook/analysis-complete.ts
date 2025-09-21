import { NextApiRequest, NextApiResponse } from 'next'
import { supabaseAdmin } from '../../../lib/supabase'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' })
  }

  try {
    const { 
      event_type, 
      analysis_id, 
      timestamp, 
      data, 
      metadata 
    } = req.body

    // Validate webhook data
    if (!analysis_id || event_type !== 'analysis.complete') {
      return res.status(400).json({ message: 'Invalid webhook data' })
    }

    console.log(`Agentic Analysis Complete: ${analysis_id}`)
    console.log('Final results summary:', {
      strategies_completed: data?.results?.agentic_metadata?.strategies_executed?.length || 0,
      quality_score: data?.results?.confidence_scores?.overall_analysis || 0,
      adaptations_made: data?.results?.autonomous_insights?.adaptation_insights?.total_adaptations || 0
    })

    // Update analysis record with final agentic results (using existing columns only)
    // Note: Store agentic final metadata in the existing analysis_results JSONB column
    const updateData = {
      analysis_status: 'completed',
      analysis_results: {
        agentic_final_metadata: {
          completion_timestamp: timestamp,
          final_quality_score: data?.results?.confidence_scores?.overall_analysis || 0,
          strategies_executed: data?.results?.agentic_metadata?.strategies_executed || [],
          adaptations_made: data?.results?.autonomous_insights?.adaptation_insights?.total_adaptations || 0,
          performance_grade: data?.results?.autonomous_insights?.data_assessment?.performance_grade || 'N/A',
          autonomous_insights_generated: Object.keys(data?.results?.autonomous_insights || {}).length
        },
        // Include any existing results
        ...(data?.results || {})
      }
    }

    const { error } = await supabaseAdmin
      .from('analyses')
      .update(updateData)
      .eq('id', analysis_id)

    if (error) {
      console.error('Failed to update analysis record:', error)
      return res.status(500).json({ message: 'Database update failed' })
    }

    // Trigger post-completion actions:
    // - Send completion notifications
    // - Update user dashboards
    // - Log analytics events
    // - Trigger follow-up workflows

    console.log(`Successfully processed analysis-complete webhook for ${analysis_id}`)

    res.status(200).json({ 
      message: 'Analysis completion webhook processed',
      analysis_id,
      agentic: true,
      quality_score: updateData.agentic_final_metadata.final_quality_score
    })

  } catch (error) {
    console.error('Webhook processing error:', error)
    res.status(500).json({ message: 'Webhook processing failed' })
  }
}