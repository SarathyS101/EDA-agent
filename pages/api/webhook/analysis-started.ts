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
    if (!analysis_id || event_type !== 'analysis.started') {
      return res.status(400).json({ message: 'Invalid webhook data' })
    }

    // Log the agentic event
    console.log(`Agentic Event: ${event_type} for analysis ${analysis_id}`)
    console.log('Event data:', data)
    console.log('Event metadata:', metadata)

    // Update analysis record (using existing columns only)
    // Note: Store agentic metadata in the existing analysis_results JSONB column
    const updateData = {
      analysis_status: 'processing',
      analysis_results: {
        agentic_metadata: {
          ...metadata,
          start_event_received: timestamp,
          csv_url: data?.csv_url,
          user_goals: data?.user_goals || []
        }
      }
    }

    // Check if analysis exists first
    const { data: existingAnalysis, error: fetchError } = await supabaseAdmin
      .from('analyses')
      .select('id')
      .eq('id', analysis_id)
      .single()

    if (fetchError || !existingAnalysis) {
      console.log(`Analysis ${analysis_id} not found in database - this is normal for webhook tests`)
      return res.status(200).json({ 
        message: 'Webhook received but analysis not found (normal for tests)',
        analysis_id,
        agentic: true
      })
    }

    const { error } = await supabaseAdmin
      .from('analyses')
      .update(updateData)
      .eq('id', analysis_id)

    if (error) {
      console.error('Failed to update analysis record:', error)
      return res.status(500).json({ message: 'Database update failed' })
    }

    // You could trigger additional actions here:
    // - Send notifications to external systems
    // - Update real-time dashboards
    // - Log to monitoring systems

    console.log(`Successfully processed analysis-started webhook for ${analysis_id}`)

    res.status(200).json({ 
      message: 'Analysis started webhook processed',
      analysis_id,
      agentic: true
    })

  } catch (error) {
    console.error('Webhook processing error:', error)
    res.status(500).json({ message: 'Webhook processing failed' })
  }
}