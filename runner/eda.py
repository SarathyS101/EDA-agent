import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import requests
import io
import os
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore')

class EdaAnalyzer:
    def __init__(self):
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Configure plotly
        pio.renderers.default = "png"
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable Python types"""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'dtype'):  # pandas/numpy dtypes including ObjectDtype
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return str(obj)
    
    def load_csv_from_url(self, csv_url: str) -> Optional[pd.DataFrame]:
        """Download and load CSV from URL"""
        try:
            response = requests.get(csv_url)
            response.raise_for_status()
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(io.StringIO(response.content.decode(encoding)))
                    print(f"Successfully loaded CSV with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
            
            raise Exception("Failed to decode CSV with any encoding")
            
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            return None
    
    def perform_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive EDA analysis"""
        results = {}
        
        # Basic info
        results['basic_info'] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Missing values analysis
        results['missing_values'] = {
            'count': df.isnull().sum().to_dict(),
            'percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        }
        
        # Data quality issues
        results['data_quality'] = self.analyze_data_quality(df)
        
        # Summary statistics
        results['summary_stats'] = self.get_summary_statistics(df)
        
        # Correlation analysis
        results['correlations'] = self.analyze_correlations(df)
        
        # Distribution analysis
        results['distributions'] = self.analyze_distributions(df)
        
        # Outlier detection
        results['outliers'] = self.detect_outliers(df)
        
        # Categorical analysis
        results['categorical_analysis'] = self.analyze_categorical_data(df)
        
        # Convert all numpy types to JSON-serializable types
        return self._convert_numpy_types(results)
    
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality issues"""
        quality_issues = {}
        
        # Duplicate rows
        quality_issues['duplicates'] = {
            'count': df.duplicated().sum(),
            'percentage': (df.duplicated().sum() / len(df) * 100).round(2)
        }
        
        # Constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        quality_issues['constant_columns'] = constant_cols
        
        # High cardinality categorical columns
        high_card_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.8:  # More than 80% unique values
                high_card_cols.append(col)
        quality_issues['high_cardinality_columns'] = high_card_cols
        
        return quality_issues
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive summary statistics"""
        summary = {}
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary['categorical'] = {}
            for col in categorical_cols:
                summary['categorical'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
        
        return summary
    
    def analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric variables"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'message': 'No numeric columns for correlation analysis'}
        
        corr_matrix = numeric_df.corr()
        
        # Find strong correlations (> 0.7 or < -0.7)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': round(corr_val, 3)
                    })
        
        return {
            'correlation_matrix': corr_matrix.round(3).to_dict(),
            'strong_correlations': strong_correlations
        }
    
    def analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze variable distributions"""
        distributions = {}
        
        # Numeric distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            distributions[col] = {
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'is_normal': abs(df[col].skew()) < 0.5  # Simple normality test
            }
        
        return distributions
    
    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        outliers = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            outliers[col] = {
                'count': int(outlier_count),
                'percentage': round(outlier_count / len(df) * 100, 2),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return outliers
    
    def analyze_categorical_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical variables"""
        categorical_analysis = {}
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            
            categorical_analysis[col] = {
                'unique_values': int(df[col].nunique()),
                'top_5_values': value_counts.head(5).to_dict(),
                'rare_values': len(value_counts[value_counts == 1]),  # Values appearing only once
                'dominant_category_percentage': round(value_counts.iloc[0] / len(df) * 100, 2) if len(value_counts) > 0 else 0
            }
        
        return categorical_analysis
    
    def generate_visualizations(self, df: pd.DataFrame, results: Dict[str, Any]) -> List[str]:
        """Generate visualization charts and save as files"""
        chart_paths = []
        
        # Create visualizations directory
        vis_dir = 'visualizations'
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Correlation heatmap
        if 'correlations' in results and 'correlation_matrix' in results['correlations']:
            corr_matrix = pd.DataFrame(results['correlations']['correlation_matrix'])
            if not corr_matrix.empty:
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                corr_path = os.path.join(vis_dir, 'correlation_heatmap.png')
                plt.savefig(corr_path, dpi=300, bbox_inches='tight')
                plt.close()
                chart_paths.append(corr_path)
        
        # 2. Missing values chart
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            plt.figure(figsize=(12, 6))
            missing_data[missing_data > 0].plot(kind='bar')
            plt.title('Missing Values by Column')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=45)
            plt.tight_layout()
            missing_path = os.path.join(vis_dir, 'missing_values.png')
            plt.savefig(missing_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths.append(missing_path)
        
        # 3. Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, min(3, len(numeric_cols)), figsize=(15, 10))
            if len(numeric_cols) == 1:
                axes = [axes]
            elif len(numeric_cols) <= 3:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols[:6]):  # Limit to 6 plots
                if i < len(axes):
                    df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_ylabel('Frequency')
            
            plt.tight_layout()
            dist_path = os.path.join(vis_dir, 'distributions.png')
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            chart_paths.append(dist_path)
        
        return chart_paths