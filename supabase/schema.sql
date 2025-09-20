-- Create custom types
CREATE TYPE analysis_status AS ENUM (
  'pending',
  'pending_payment', 
  'paid',
  'processing',
  'completed',
  'failed'
);

-- Create analyses table
CREATE TABLE public.analyses (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  csv_filename TEXT NOT NULL,
  csv_url TEXT NOT NULL,
  pdf_url TEXT,
  analysis_status analysis_status DEFAULT 'pending' NOT NULL,
  payment_intent_id TEXT,
  analysis_results JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Create indexes for better performance
CREATE INDEX idx_analyses_user_id ON public.analyses(user_id);
CREATE INDEX idx_analyses_status ON public.analyses(analysis_status);
CREATE INDEX idx_analyses_created_at ON public.analyses(created_at DESC);
CREATE INDEX idx_analyses_payment_intent ON public.analyses(payment_intent_id) WHERE payment_intent_id IS NOT NULL;

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for updated_at
CREATE TRIGGER handle_analyses_updated_at
  BEFORE UPDATE ON public.analyses
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_updated_at();

-- Row Level Security (RLS) policies
ALTER TABLE public.analyses ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own analyses
CREATE POLICY "Users can view own analyses" ON public.analyses
  FOR SELECT USING (auth.uid() = user_id);

-- Policy: Users can insert their own analyses
CREATE POLICY "Users can insert own analyses" ON public.analyses
  FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Policy: Users can update their own analyses
CREATE POLICY "Users can update own analyses" ON public.analyses
  FOR UPDATE USING (auth.uid() = user_id);

-- Policy: Service role can do everything (for API operations)
CREATE POLICY "Service role can manage all analyses" ON public.analyses
  FOR ALL USING (
    current_setting('request.jwt.claims', true)::json->>'role' = 'service_role'
  );

-- Create storage buckets
INSERT INTO storage.buckets (id, name, public) VALUES 
  ('csv-files', 'csv-files', true),
  ('pdf-reports', 'pdf-reports', true);

-- Storage policies for csv-files bucket
CREATE POLICY "Users can upload their own CSV files" ON storage.objects
  FOR INSERT WITH CHECK (
    bucket_id = 'csv-files' AND
    auth.uid()::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Users can view their own CSV files" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'csv-files' AND
    auth.uid()::text = (storage.foldername(name))[1]
  );

-- Storage policies for pdf-reports bucket
CREATE POLICY "Users can view their own PDF reports" ON storage.objects
  FOR SELECT USING (
    bucket_id = 'pdf-reports' AND
    auth.uid()::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Service role can manage PDF reports" ON storage.objects
  FOR ALL USING (
    bucket_id = 'pdf-reports' AND
    current_setting('request.jwt.claims', true)::json->>'role' = 'service_role'
  );

-- Create a view for user statistics (optional)
CREATE VIEW public.user_analysis_stats AS
SELECT 
  user_id,
  COUNT(*) as total_analyses,
  COUNT(*) FILTER (WHERE analysis_status = 'completed') as completed_analyses,
  COUNT(*) FILTER (WHERE analysis_status IN ('processing', 'paid', 'pending')) as pending_analyses,
  MAX(created_at) as last_analysis_date
FROM public.analyses
GROUP BY user_id;

-- Grant access to the view
GRANT SELECT ON public.user_analysis_stats TO authenticated;

-- RLS for the view
ALTER VIEW public.user_analysis_stats SET (security_invoker = true);

-- Create function to clean up old temporary files (run periodically)
CREATE OR REPLACE FUNCTION public.cleanup_old_analyses()
RETURNS void AS $$
BEGIN
  -- Delete analyses older than 30 days that failed or were never paid
  DELETE FROM public.analyses 
  WHERE created_at < NOW() - INTERVAL '30 days'
    AND analysis_status IN ('failed', 'pending', 'pending_payment');
    
  -- Log cleanup
  RAISE NOTICE 'Cleaned up old analyses';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create a function to get analysis with user validation
CREATE OR REPLACE FUNCTION public.get_user_analysis(analysis_uuid UUID)
RETURNS TABLE(
  id UUID,
  csv_filename TEXT,
  analysis_status analysis_status,
  pdf_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE,
  analysis_results JSONB
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    a.id,
    a.csv_filename,
    a.analysis_status,
    a.pdf_url,
    a.created_at,
    a.analysis_results
  FROM public.analyses a
  WHERE a.id = analysis_uuid AND a.user_id = auth.uid();
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

GRANT EXECUTE ON FUNCTION public.get_user_analysis(UUID) TO authenticated;