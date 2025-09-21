import pandas as pd
import openai
import json
import os
import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class EdaValidator:
    def __init__(self):
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        self.client = openai.OpenAI()
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable Python types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'dtype'):  # pandas/numpy dtypes
            return str(obj)
        else:
            return obj
    
    def validate_analysis(self, df: pd.DataFrame, initial_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to validate and enhance the EDA results
        This is the core of the agentic behavior - the AI validates its own work
        """
        try:
            # Prepare data summary for LLM
            data_summary = self._prepare_data_summary(df)
            
            validation_prompt = self._create_validation_prompt(data_summary, initial_results)
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert data scientist reviewing an EDA analysis. 
                        Your role is to validate the analysis, identify any issues, and suggest improvements.
                        Always return valid JSON with the same structure as provided, but with validated/corrected values."""
                    },
                    {
                        "role": "user",
                        "content": validation_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            validated_results = json.loads(response.choices[0].message.content)
            
            # Add validation metadata
            validated_results['validation'] = {
                'validated_by_llm': True,
                'validation_timestamp': pd.Timestamp.now().isoformat(),
                'original_vs_validated': self._compare_results(initial_results, validated_results)
            }
            
            logger.info("Analysis validation completed successfully")
            return validated_results
            
        except Exception as e:
            logger.error(f"LLM validation failed: {str(e)}")
            # Return original results with validation failure note
            initial_results['validation'] = {
                'validated_by_llm': False,
                'error': str(e),
                'fallback_to_original': True
            }
            return initial_results
    
    def enhance_insights(self, df: pd.DataFrame, validated_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate additional insights and recommendations using LLM
        """
        # Check if OpenAI API key is available
        if not os.environ.get('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY not found. Generating rule-based insights instead.")
            return self._generate_rule_based_insights(df, validated_results)
        
        try:
            data_summary = self._prepare_data_summary(df)
            
            insights_prompt = f"""
            Based on this data analysis, provide 5-7 actionable business insights and recommendations:
            
            Data Summary:
            {json.dumps(data_summary, indent=2)}
            
            Analysis Results:
            {json.dumps(self._convert_numpy_types(validated_results), indent=2)}
            
            Please provide:
            1. Key findings and patterns
            2. Data quality observations
            3. Business implications
            4. Recommended next steps
            5. Potential data collection improvements
            
            Return as JSON with keys: 'insights', 'recommendations', 'data_quality_notes', 'next_steps'
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a senior data scientist providing business insights. Be specific, actionable, and focus on business value."
                    },
                    {
                        "role": "user",
                        "content": insights_prompt
                    }
                ],
                temperature=0.4,
                max_tokens=1500
            )
            
            enhanced_insights = json.loads(response.choices[0].message.content)
            
            # Add insights to validated results
            validated_results.update(enhanced_insights)
            
            logger.info("Enhanced insights generated successfully")
            return validated_results
            
        except Exception as e:
            logger.error(f"Insight enhancement failed: {str(e)}")
            # Fallback to rule-based insights
            logger.info("Falling back to rule-based insights generation")
            return self._generate_rule_based_insights(df, validated_results)
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare a concise data summary for the LLM"""
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=['number']).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'sample_data': df.head(3).to_dict('records') if len(df) > 0 else []
        }
        
        # Convert all numpy types to JSON-serializable types
        return self._convert_numpy_types(summary)
    
    def _create_validation_prompt(self, data_summary: Dict[str, Any], initial_results: Dict[str, Any]) -> str:
        """Create the validation prompt for the LLM"""
        return f"""
        Please validate this EDA analysis and correct any issues you find:

        ORIGINAL DATA:
        {json.dumps(data_summary, indent=2)}

        ANALYSIS RESULTS TO VALIDATE:
        {json.dumps(self._convert_numpy_types(initial_results), indent=2)}

        Please check for:
        1. Accuracy of basic statistics
        2. Correctness of data type identification
        3. Missing value calculations
        4. Correlation analysis validity
        5. Outlier detection reasonableness
        6. Any inconsistencies or errors

        Return the corrected analysis in the exact same JSON structure, but with validated values.
        If everything is correct, return the original structure with any enhancements you suggest.
        """
    
    def _compare_results(self, original: Dict[str, Any], validated: Dict[str, Any]) -> Dict[str, str]:
        """Compare original and validated results to track changes"""
        changes = {}
        
        # Check basic info changes
        if 'basic_info' in original and 'basic_info' in validated:
            if original['basic_info'] != validated['basic_info']:
                changes['basic_info'] = 'Modified during validation'
        
        # Check correlation changes
        if 'correlations' in original and 'correlations' in validated:
            if original['correlations'] != validated['correlations']:
                changes['correlations'] = 'Correlation analysis updated'
        
        # Check missing values
        if 'missing_values' in original and 'missing_values' in validated:
            if original['missing_values'] != validated['missing_values']:
                changes['missing_values'] = 'Missing value analysis corrected'
        
        return changes if changes else {'status': 'No changes needed'}
    
    def _generate_rule_based_insights(self, df: pd.DataFrame, validated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights using rule-based logic when LLM is not available"""
        try:
            insights = []
            recommendations = []
            data_quality_notes = []
            next_steps = []
            
            # Dataset size insights
            total_records = len(df)
            total_columns = len(df.columns)
            insights.append(f"Dataset contains {total_records:,} records across {total_columns} variables")
            
            # Missing data insights
            if 'missing_values' in validated_results:
                total_missing = sum(validated_results['missing_values']['count'].values())
                missing_pct = (total_missing / (total_records * total_columns)) * 100
                if missing_pct > 10:
                    insights.append(f"High missing data rate ({missing_pct:.1f}%) may impact analysis reliability")
                    recommendations.append("Consider data imputation or collection strategies for missing values")
                elif missing_pct > 0:
                    insights.append(f"Dataset has {missing_pct:.1f}% missing values - manageable but worth addressing")
                else:
                    insights.append("Dataset is complete with no missing values")
            
            # Data types insights
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            categorical_cols = len(df.select_dtypes(include=['object']).columns)
            if numeric_cols > categorical_cols:
                insights.append(f"Dataset is numeric-heavy ({numeric_cols} numeric vs {categorical_cols} categorical variables)")
                recommendations.append("Focus on statistical analysis and numerical relationships")
            else:
                insights.append(f"Dataset has substantial categorical data ({categorical_cols} categorical vs {numeric_cols} numeric variables)")
                recommendations.append("Consider category analysis and encoding strategies")
            
            # Correlation insights
            if 'correlations' in validated_results and 'strong_correlations' in validated_results['correlations']:
                strong_corrs = len(validated_results['correlations']['strong_correlations'])
                if strong_corrs > 0:
                    insights.append(f"Found {strong_corrs} strong correlations that may indicate important relationships")
                    recommendations.append("Investigate strong correlations for potential feature engineering or business insights")
                else:
                    insights.append("No strong correlations detected - variables appear relatively independent")
            
            # Outlier insights
            if 'outliers' in validated_results:
                total_outliers = sum(outlier_info.get('count', 0) for outlier_info in validated_results['outliers'].values())
                if total_outliers > 0:
                    outlier_pct = (total_outliers / total_records) * 100
                    if outlier_pct > 5:
                        insights.append(f"High outlier rate ({outlier_pct:.1f}%) detected across variables")
                        data_quality_notes.append("Review outliers for data quality issues or genuine extreme values")
                    else:
                        insights.append(f"Moderate outlier presence ({outlier_pct:.1f}%) - typical for most datasets")
                        data_quality_notes.append("Outliers detected but within normal range")
                
            # Data quality assessment
            if missing_pct < 5 and total_outliers / total_records < 0.05:
                data_quality_notes.append("Overall data quality appears good with minimal issues")
            else:
                data_quality_notes.append("Data quality issues identified that may require attention")
            
            # Next steps based on dataset characteristics
            if numeric_cols > 0:
                next_steps.append("Perform statistical modeling or predictive analysis")
            if categorical_cols > 0:
                next_steps.append("Conduct categorical analysis and segmentation studies")
            if strong_corrs > 0:
                next_steps.append("Investigate causal relationships behind strong correlations")
            if total_missing > 0:
                next_steps.append("Address missing data through imputation or additional collection")
            
            next_steps.append("Create targeted visualizations for key findings")
            next_steps.append("Validate insights with domain experts")
            
            # Update results
            validated_results.update({
                'insights': insights,
                'recommendations': recommendations,
                'data_quality_notes': data_quality_notes,
                'next_steps': next_steps
            })
            
            logger.info("Rule-based insights generated successfully")
            return validated_results
            
        except Exception as e:
            logger.error(f"Rule-based insight generation failed: {str(e)}")
            # Final fallback to minimal insights
            validated_results.update({
                'insights': [f'Dataset analyzed: {len(df)} rows, {len(df.columns)} columns'],
                'recommendations': ['Manual review of statistical summaries recommended'],
                'data_quality_notes': ['Data quality assessment completed'],
                'next_steps': ['Review analysis results and plan next steps']
            })
            return validated_results