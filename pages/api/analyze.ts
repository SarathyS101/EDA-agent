import { NextApiRequest, NextApiResponse } from 'next'
import { supabaseAdmin } from '@/lib/supabase'
import { spawn } from 'child_process'
import path from 'path'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' })
  }

  try {
    const { analysisId, userId, skipPayment } = req.body

    if (!analysisId || !userId) {
      return res.status(400).json({ message: 'Missing required fields' })
    }

    // Get analysis record
    const { data: analysis, error: fetchError } = await supabaseAdmin
      .from('analyses')
      .select('*')
      .eq('id', analysisId)
      .eq('user_id', userId)
      .single()

    if (fetchError || !analysis) {
      return res.status(404).json({ message: 'Analysis not found' })
    }

    // Check if payment is completed (skip this check for local testing)
    if (!skipPayment && analysis.analysis_status !== 'paid') {
      return res.status(400).json({ message: 'Payment required before analysis' })
    }

    // If skipPayment is true, mark as paid for local testing
    if (skipPayment && analysis.analysis_status === 'pending') {
      await supabaseAdmin
        .from('analyses')
        .update({ analysis_status: 'paid' })
        .eq('id', analysisId)
    }

    // Update status to processing
    await supabaseAdmin
      .from('analyses')
      .update({ analysis_status: 'processing' })
      .eq('id', analysisId)

    // Trigger Python analysis script
    const pythonScriptPath = path.join(process.cwd(), 'runner', 'agent.py')
    const pythonProcess = spawn('python3', [pythonScriptPath, analysis.csv_url, analysisId, userId])

    pythonProcess.stdout.on('data', (data) => {
      // Python output handled internally
    })

    pythonProcess.stderr.on('data', (data) => {
      // Python errors handled internally  
    })

    pythonProcess.on('close', async (code) => {
      if (code !== 0) {
        await supabaseAdmin
          .from('analyses')
          .update({ analysis_status: 'failed' })
          .eq('id', analysisId)
      }
    })

    res.status(200).json({ 
      message: 'Analysis started',
      analysisId,
      status: 'processing'
    })

  } catch (error) {
    console.error('Analysis error:', error)
    res.status(500).json({ message: 'Failed to start analysis' })
  }
}

// Alternative endpoint to check analysis status
export async function checkAnalysisStatus(req: NextApiRequest, res: NextApiResponse) {
  const { analysisId, userId } = req.query

  try {
    const { data: analysis, error } = await supabaseAdmin
      .from('analyses')
      .select('*')
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
      results: analysis.analysis_results
    })

  } catch (error) {
    console.error('Status check error:', error)
    res.status(500).json({ message: 'Failed to check analysis status' })
  }
}