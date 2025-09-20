#!/usr/bin/env python3
"""
Test script to verify Python EDA pipeline works locally
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the runner directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'runner'))

try:
    from eda import EdaAnalyzer
    from validator import EdaValidator
    from pdf_generator import PdfGenerator
    print("✅ All Python modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def create_sample_data():
    """Create a sample CSV file for testing"""
    print("📊 Creating sample data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'score': np.random.beta(2, 5, n_samples) * 100,
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'is_active': np.random.choice([True, False], n_samples),
        'rating': np.random.uniform(1, 5, n_samples)
    }
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    data['income'][missing_indices] = np.nan
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = 'test_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Sample data created: {csv_path}")
    
    return df, csv_path

def test_eda_analyzer():
    """Test the EDA analyzer"""
    print("\n🔍 Testing EDA Analyzer...")
    
    df, csv_path = create_sample_data()
    analyzer = EdaAnalyzer()
    
    # Test EDA analysis
    results = analyzer.perform_eda(df)
    
    print("✅ EDA analysis completed")
    print(f"   - Basic info: {bool(results.get('basic_info'))}")
    print(f"   - Missing values: {bool(results.get('missing_values'))}")
    print(f"   - Summary stats: {bool(results.get('summary_stats'))}")
    print(f"   - Correlations: {bool(results.get('correlations'))}")
    
    # Test visualizations
    try:
        viz_paths = analyzer.generate_visualizations(df, results)
        print(f"✅ Generated {len(viz_paths)} visualizations")
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")
    
    # Clean up
    if os.path.exists(csv_path):
        os.remove(csv_path)
    
    return results

def test_without_llm():
    """Test the system without LLM validation (for cases where API key is not set)"""
    print("\n🤖 Testing without LLM validation...")
    
    df, csv_path = create_sample_data()
    analyzer = EdaAnalyzer()
    
    # Get basic results
    results = analyzer.perform_eda(df)
    
    # Create PDF without LLM validation
    pdf_generator = PdfGenerator()
    
    try:
        pdf_path = pdf_generator.create_report(df, results, "test_report_no_llm.pdf")
        print(f"✅ PDF generated without LLM: {pdf_path}")
        
        # Clean up
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
    
    # Clean up CSV
    if os.path.exists(csv_path):
        os.remove(csv_path)

def test_with_llm():
    """Test with LLM validation if OpenAI API key is available"""
    print("\n🤖 Testing with LLM validation...")
    
    if not os.environ.get('OPENAI_API_KEY'):
        print("⚠️  OPENAI_API_KEY not found, skipping LLM tests")
        return
    
    df, csv_path = create_sample_data()
    analyzer = EdaAnalyzer()
    validator = EdaValidator()
    pdf_generator = PdfGenerator()
    
    try:
        # Get initial results
        initial_results = analyzer.perform_eda(df)
        
        # Validate with LLM
        validated_results = validator.validate_analysis(df, initial_results)
        print("✅ LLM validation completed")
        
        # Enhance insights
        enhanced_results = validator.enhance_insights(df, validated_results)
        print("✅ Enhanced insights generated")
        
        # Generate PDF
        pdf_path = pdf_generator.create_report(df, enhanced_results, "test_report_with_llm.pdf")
        print(f"✅ PDF generated with LLM: {pdf_path}")
        
        # Clean up
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            
    except Exception as e:
        print(f"❌ LLM testing failed: {e}")
    
    # Clean up CSV
    if os.path.exists(csv_path):
        os.remove(csv_path)

def main():
    print("🧪 Testing Agentic EDA Python Pipeline")
    print("=" * 50)
    
    # Test basic EDA
    try:
        results = test_eda_analyzer()
        print("✅ Basic EDA tests passed")
    except Exception as e:
        print(f"❌ Basic EDA tests failed: {e}")
        return
    
    # Test without LLM
    test_without_llm()
    
    # Test with LLM if available
    test_with_llm()
    
    # Clean up visualization directory
    viz_dir = 'visualizations'
    if os.path.exists(viz_dir):
        import shutil
        shutil.rmtree(viz_dir)
    
    print("\n🎉 Testing completed!")
    print("\nTo test the full application:")
    print("1. Make sure your .env file has all required API keys")
    print("2. Run 'npm run dev' to start the Next.js server")
    print("3. Visit http://localhost:3000")

if __name__ == "__main__":
    main()