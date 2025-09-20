import { useState, useEffect } from 'react'
import { useRouter } from 'next/router'
import { BarChart3, Download, Clock, CheckCircle, XCircle, FileText, Upload, LogOut } from 'lucide-react'
import axios from 'axios'
import { formatDate } from '@/lib/utils'
import { useAuth } from '@/contexts/AuthContext'
import { ProtectedRoute } from '@/components/ProtectedRoute'

interface Analysis {
  id: string
  csv_filename: string
  analysis_status: 'pending' | 'processing' | 'completed' | 'failed' | 'pending_payment' | 'paid'
  pdf_url?: string
  created_at: string
  updated_at: string
}

interface UserStats {
  totalAnalyses: number
  completedAnalyses: number
  pendingAnalyses: number
}

function DashboardContent() {
  const [analyses, setAnalyses] = useState<Analysis[]>([])
  const [stats, setStats] = useState<UserStats>({ totalAnalyses: 0, completedAnalyses: 0, pendingAnalyses: 0 })
  const [loading, setLoading] = useState(true)
  const { user, signOut } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (user) {
      fetchAnalyses()
      
      // Check if we're returning from a successful payment
      const { session_id } = router.query
      if (session_id) {
        // Refresh the analyses to get updated payment status
        setTimeout(() => fetchAnalyses(), 2000)
      }
    }
  }, [user, router.query])

  const fetchAnalyses = async () => {
    if (!user) return

    try {
      const response = await axios.get(`/api/history?userId=${user.id}`)
      setAnalyses(response.data.analyses || [])
      setStats(response.data.stats || { totalAnalyses: 0, completedAnalyses: 0, pendingAnalyses: 0 })
    } catch (error) {
      console.error('Failed to fetch analyses:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'processing':
      case 'paid':
        return <Clock className="h-5 w-5 text-blue-500" />
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <Clock className="h-5 w-5 text-gray-500" />
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'pending_payment':
        return 'Payment Required'
      case 'paid':
        return 'Processing'
      case 'processing':
        return 'Analyzing'
      case 'completed':
        return 'Completed'
      case 'failed':
        return 'Failed'
      default:
        return 'Pending'
    }
  }

  const handleSignOut = async () => {
    await signOut()
  }

  const downloadPDF = (pdfUrl: string, filename: string) => {
    const link = document.createElement('a')
    link.href = pdfUrl
    link.download = `${filename.replace('.csv', '')}_analysis.pdf`
    link.target = '_blank'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }


  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <BarChart3 className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">Agentic EDA</span>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-700">Hello, {user?.email}</span>
              <button
                onClick={() => router.push('/')}
                className="btn-secondary"
              >
                <Upload className="h-4 w-4 mr-2" />
                New Analysis
              </button>
              <button
                onClick={handleSignOut}
                className="btn-secondary"
              >
                <LogOut className="h-4 w-4 mr-2" />
                Sign Out
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="card">
            <div className="flex items-center">
              <div className="p-3 bg-blue-100 rounded-lg">
                <FileText className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Analyses</p>
                <p className="text-2xl font-bold text-gray-900">{stats.totalAnalyses}</p>
              </div>
            </div>
          </div>
          
          <div className="card">
            <div className="flex items-center">
              <div className="p-3 bg-green-100 rounded-lg">
                <CheckCircle className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Completed</p>
                <p className="text-2xl font-bold text-gray-900">{stats.completedAnalyses}</p>
              </div>
            </div>
          </div>
          
          <div className="card">
            <div className="flex items-center">
              <div className="p-3 bg-yellow-100 rounded-lg">
                <Clock className="h-6 w-6 text-yellow-600" />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">In Progress</p>
                <p className="text-2xl font-bold text-gray-900">{stats.pendingAnalyses}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Analysis History */}
        <div className="card">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-gray-900">Analysis History</h2>
            <button
              onClick={fetchAnalyses}
              className="btn-secondary text-sm"
            >
              Refresh
            </button>
          </div>

          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
              <p className="text-gray-600 mt-2">Loading...</p>
            </div>
          ) : analyses.length === 0 ? (
            <div className="text-center py-8">
              <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-600">No analyses yet. Upload your first CSV file to get started!</p>
              <button
                onClick={() => router.push('/')}
                className="btn-primary mt-4"
              >
                Upload CSV File
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      File Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Created
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {analyses.map((analysis) => (
                    <tr key={analysis.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <FileText className="h-5 w-5 text-gray-400 mr-3" />
                          <div>
                            <p className="text-sm font-medium text-gray-900">
                              {analysis.csv_filename}
                            </p>
                            <p className="text-sm text-gray-500">ID: {analysis.id.slice(0, 8)}...</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          {getStatusIcon(analysis.analysis_status)}
                          <span className="ml-2 text-sm text-gray-900">
                            {getStatusText(analysis.analysis_status)}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDate(analysis.created_at)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                        {analysis.analysis_status === 'completed' && analysis.pdf_url ? (
                          <button
                            onClick={() => downloadPDF(analysis.pdf_url!, analysis.csv_filename)}
                            className="text-blue-600 hover:text-blue-900 inline-flex items-center"
                          >
                            <Download className="h-4 w-4 mr-1" />
                            Download PDF
                          </button>
                        ) : analysis.analysis_status === 'pending_payment' ? (
                          <button
                            onClick={() => router.push(`/payment?analysisId=${analysis.id}`)}
                            className="text-blue-600 hover:text-blue-900"
                          >
                            Complete Payment
                          </button>
                        ) : analysis.analysis_status === 'processing' || analysis.analysis_status === 'paid' ? (
                          <span className="text-blue-600">Processing...</span>
                        ) : analysis.analysis_status === 'failed' ? (
                          <span className="text-red-600">Analysis Failed</span>
                        ) : (
                          <span className="text-gray-500">Pending</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function Dashboard() {
  return (
    <ProtectedRoute>
      <DashboardContent />
    </ProtectedRoute>
  )
}