import { NextApiRequest, NextApiResponse } from 'next'
import formidable from 'formidable'
import fs from 'fs'
import { supabaseAdmin } from '@/lib/supabase'
import { generateSecureFilename, isValidCSV } from '@/lib/utils'

export const config = {
  api: {
    bodyParser: false,
  },
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' })
  }

  try {
    const form = formidable({
      maxFileSize: 10 * 1024 * 1024, // 10MB limit
    })

    const [fields, files] = await form.parse(req)
    const userId = Array.isArray(fields.userId) ? fields.userId[0] : fields.userId
    const file = Array.isArray(files.csv) ? files.csv[0] : files.csv

    if (!file || !userId) {
      return res.status(400).json({ message: 'Missing file or user ID' })
    }

    if (!isValidCSV(file.originalFilename || '')) {
      return res.status(400).json({ message: 'Please upload a valid CSV file' })
    }

    // Generate secure filename
    const secureFilename = generateSecureFilename(file.originalFilename || 'upload.csv')
    const fileBuffer = fs.readFileSync(file.filepath)

    // Upload to Supabase Storage
    const { data: storageData, error: storageError } = await supabaseAdmin.storage
      .from('csv-files')
      .upload(`${userId}/${secureFilename}`, fileBuffer, {
        contentType: 'text/csv',
        upsert: false
      })

    if (storageError) {
      console.error('Storage error:', storageError)
      return res.status(500).json({ message: 'Failed to upload file' })
    }

    // Get public URL
    const { data: urlData } = supabaseAdmin.storage
      .from('csv-files')
      .getPublicUrl(`${userId}/${secureFilename}`)

    // Create analysis record in database
    const { data: analysisData, error: dbError } = await supabaseAdmin
      .from('analyses')
      .insert({
        user_id: userId,
        csv_filename: file.originalFilename,
        csv_url: urlData.publicUrl,
        analysis_status: 'pending'
      })
      .select()
      .single()

    if (dbError) {
      console.error('Database error:', dbError)
      return res.status(500).json({ message: 'Failed to create analysis record' })
    }

    // Clean up temp file
    fs.unlinkSync(file.filepath)

    res.status(200).json({
      message: 'File uploaded successfully',
      analysisId: analysisData.id,
      csvUrl: urlData.publicUrl
    })

  } catch (error) {
    console.error('Upload error:', error)
    res.status(500).json({ message: 'Internal server error' })
  }
}