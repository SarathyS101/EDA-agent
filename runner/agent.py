#!/usr/bin/env python3

import sys
import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv

# Import our modules
from eda import EdaAnalyzer
from validator import EdaValidator
from pdf_generator import PdfGenerator

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgenticEdaAgent:
    def __init__(self):
        self.supabase: Client = create_client(
            os.environ.get('NEXT_PUBLIC_SUPABASE_URL'),
            os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
        )
        self.analyzer = EdaAnalyzer()
        self.validator = EdaValidator()
        self.pdf_generator = PdfGenerator()
    
    def run_analysis(self, csv_url: str, analysis_id: str, user_id: str) -> Dict[str, Any]:
        """
        Main agentic analysis workflow:
        1. Download and analyze CSV
        2. Generate preliminary EDA results
        3. Validate results with LLM
        4. Generate additional insights
        5. Create PDF report
        6. Upload and update database
        """
        try:
            logger.info(f"Starting analysis for {analysis_id}")
            
            # Step 1: Download and load CSV
            logger.info("Step 1: Loading CSV data...")
            df = self.analyzer.load_csv_from_url(csv_url)
            if df is None:
                raise Exception("Failed to load CSV data")
            
            # Step 2: Run initial EDA
            logger.info("Step 2: Running initial EDA...")
            initial_results = self.analyzer.perform_eda(df)
            
            # Step 3: Validate with LLM
            logger.info("Step 3: Validating results with LLM...")
            validated_results = self.validator.validate_analysis(df, initial_results)
            
            # Step 4: Generate additional insights
            logger.info("Step 4: Generating additional insights...")
            enhanced_results = self.validator.enhance_insights(df, validated_results)
            
            # Step 5: Generate visualizations
            logger.info("Step 5: Creating visualizations...")
            chart_paths = self.analyzer.generate_visualizations(df, enhanced_results)
            enhanced_results['visualizations'] = chart_paths
            
            # Step 6: Generate PDF report
            logger.info("Step 6: Generating PDF report...")
            pdf_path = self.pdf_generator.create_report(
                df, enhanced_results, f"eda_report_{analysis_id}.pdf"
            )
            
            # Step 7: Upload PDF to Supabase
            logger.info("Step 7: Uploading PDF to Supabase...")
            pdf_url = self.upload_pdf_to_supabase(pdf_path, user_id, analysis_id)
            
            # Step 8: Update database with results
            logger.info("Step 8: Updating database...")
            self.update_analysis_record(analysis_id, enhanced_results, pdf_url)
            
            logger.info(f"Analysis completed successfully for {analysis_id}")
            return {
                'status': 'completed',
                'pdf_url': pdf_url,
                'results': enhanced_results
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for {analysis_id}: {str(e)}")
            self.update_analysis_record(analysis_id, None, None, 'failed')
            return {'status': 'failed', 'error': str(e)}
    
    def upload_pdf_to_supabase(self, pdf_path: str, user_id: str, analysis_id: str) -> str:
        """Upload PDF to Supabase storage and return public URL"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            storage_path = f"{user_id}/reports/{analysis_id}.pdf"
            
            result = self.supabase.storage.from_('pdf-reports').upload(
                storage_path, pdf_data, {'content-type': 'application/pdf'}
            )
            
            # Always try to get the public URL - upload likely succeeded
            try:
                url_response = self.supabase.storage.from_('pdf-reports').get_public_url(storage_path)
                
                # Handle different response formats
                if isinstance(url_response, str):
                    public_url = url_response
                elif hasattr(url_response, 'publicURL'):
                    public_url = url_response.publicURL
                elif isinstance(url_response, dict):
                    public_url = url_response.get('publicURL') or url_response.get('publicUrl')
                else:
                    # Construct the URL manually as fallback
                    base_url = os.environ.get('NEXT_PUBLIC_SUPABASE_URL')
                    public_url = f"{base_url}/storage/v1/object/public/pdf-reports/{storage_path}"
                
                if not public_url:
                    raise Exception("Could not generate public URL")
                    
                return public_url
                
            except Exception as e:
                logger.error(f"Failed to get public URL: {str(e)}")
                # Fallback: construct URL manually
                base_url = os.environ.get('NEXT_PUBLIC_SUPABASE_URL')
                return f"{base_url}/storage/v1/object/public/pdf-reports/{storage_path}"
                
        except Exception as e:
            logger.error(f"Failed to upload PDF: {str(e)}")
            raise
        finally:
            # Clean up local PDF file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
    
    def update_analysis_record(self, analysis_id: str, results: Dict[str, Any] = None, 
                             pdf_url: str = None, status: str = 'completed'):
        """Update the analysis record in the database"""
        try:
            update_data = {
                'analysis_status': status,
                'updated_at': 'now()'
            }
            
            if results:
                update_data['analysis_results'] = results
            if pdf_url:
                update_data['pdf_url'] = pdf_url
            
            result = self.supabase.table('analyses').update(update_data).eq('id', analysis_id).execute()
            
            if not result.data:
                raise Exception("Failed to update analysis record")
                
        except Exception as e:
            logger.error(f"Failed to update database: {str(e)}")
            raise

def main():
    if len(sys.argv) != 4:
        print("Usage: python agent.py <csv_url> <analysis_id> <user_id>")
        sys.exit(1)
    
    csv_url = sys.argv[1]
    analysis_id = sys.argv[2] 
    user_id = sys.argv[3]
    
    agent = AgenticEdaAgent()
    result = agent.run_analysis(csv_url, analysis_id, user_id)
    
    # Convert numpy types before printing JSON
    def convert_numpy_types(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, 'dtype'):  # numpy/pandas types
            if 'int' in str(obj.dtype):
                return int(obj)
            elif 'float' in str(obj.dtype):
                return float(obj)
            elif 'bool' in str(obj.dtype):
                return bool(obj)
            else:
                return str(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return str(obj)
    
    print(json.dumps(convert_numpy_types(result)))
    sys.exit(0 if result['status'] == 'completed' else 1)

if __name__ == "__main__":
    main()