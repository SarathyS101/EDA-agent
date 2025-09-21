#!/usr/bin/env python3

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataProfiler:
    """
    Intelligent data profiler that categorizes datasets and identifies analysis opportunities.
    This is the foundation for agentic behavior - understanding what we're working with.
    """
    
    def __init__(self):
        self.profile_cache = {}
    
    def profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive dataset profiling that forms the basis for agentic decision-making.
        Returns a profile that guides the planning agent.
        """
        try:
            profile = {
                'dataset_id': self._generate_dataset_id(df),
                'basic_stats': self._get_basic_stats(df),
                'column_types': self._analyze_column_types(df),
                'data_quality': self._assess_data_quality(df),
                'patterns': self._detect_patterns(df),
                'complexity': self._assess_complexity(df),
                'analysis_opportunities': self._identify_opportunities(df),
                'recommended_strategies': []  # Will be filled by planner
            }
            
            # Cache the profile for future reference
            self.profile_cache[profile['dataset_id']] = profile
            
            logger.info(f"Dataset profiled: {profile['basic_stats']['shape']}, "
                       f"complexity: {profile['complexity']['level']}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to profile dataset: {str(e)}")
            return self._get_fallback_profile(df)
    
    def _generate_dataset_id(self, df: pd.DataFrame) -> str:
        """Generate unique identifier for dataset based on structure and content"""
        shape_str = f"{df.shape[0]}x{df.shape[1]}"
        columns_hash = str(hash(tuple(sorted(df.columns))))[:8]
        return f"dataset_{shape_str}_{columns_hash}"
    
    def _get_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract fundamental dataset statistics"""
        return {
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'column_names': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'size_category': self._categorize_dataset_size(df)
        }
    
    def _categorize_dataset_size(self, df: pd.DataFrame) -> str:
        """Categorize dataset size for strategy selection"""
        rows, cols = df.shape
        
        if rows < 1000 and cols < 10:
            return 'small'
        elif rows < 10000 and cols < 50:
            return 'medium'
        elif rows < 100000 and cols < 100:
            return 'large'
        else:
            return 'very_large'
    
    def _analyze_column_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze column types and identify special patterns"""
        analysis = {
            'numerical': [],
            'categorical': [],
            'datetime': [],
            'text': [],
            'binary': [],
            'mixed': [],
            'potential_ids': [],
            'potential_targets': []
        }
        
        for col in df.columns:
            col_type = self._classify_column(df[col], col)
            analysis[col_type['primary_type']].append({
                'name': col,
                'subtype': col_type.get('subtype'),
                'confidence': col_type.get('confidence', 1.0),
                'special_properties': col_type.get('special_properties', [])
            })
        
        return analysis
    
    def _classify_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Intelligently classify individual columns"""
        # Check for datetime patterns
        if self._is_datetime_column(series):
            return {
                'primary_type': 'datetime',
                'subtype': self._get_datetime_subtype(series),
                'confidence': 0.95
            }
        
        # Check for categorical data
        if self._is_categorical_column(series):
            return {
                'primary_type': 'categorical',
                'subtype': self._get_categorical_subtype(series),
                'confidence': 0.9,
                'special_properties': self._get_categorical_properties(series)
            }
        
        # Check for numerical data
        if pd.api.types.is_numeric_dtype(series):
            return {
                'primary_type': 'numerical',
                'subtype': self._get_numerical_subtype(series),
                'confidence': 0.95,
                'special_properties': self._get_numerical_properties(series)
            }
        
        # Check for potential IDs
        if self._is_potential_id(series, col_name):
            return {
                'primary_type': 'potential_ids',
                'confidence': 0.8,
                'special_properties': ['unique_identifier']
            }
        
        # Check for text data
        if pd.api.types.is_string_dtype(series):
            return {
                'primary_type': 'text',
                'subtype': self._get_text_subtype(series),
                'confidence': 0.85
            }
        
        return {'primary_type': 'mixed', 'confidence': 0.5}
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        """Detect datetime columns with various formats"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Try to parse as datetime
        if pd.api.types.is_string_dtype(series):
            try:
                pd.to_datetime(series.dropna().head(100))
                return True
            except:
                return False
        
        return False
    
    def _get_datetime_subtype(self, series: pd.Series) -> str:
        """Determine datetime subtype"""
        try:
            dt_series = pd.to_datetime(series.dropna())
            time_range = dt_series.max() - dt_series.min()
            
            if time_range.days > 365:
                return 'long_term_timeseries'
            elif time_range.days > 30:
                return 'medium_term_timeseries'
            else:
                return 'short_term_timeseries'
        except:
            return 'unknown_datetime'
    
    def _is_categorical_column(self, series: pd.Series) -> bool:
        """Detect categorical columns"""
        if pd.api.types.is_categorical_dtype(series):
            return True
        
        # Check if it's likely categorical based on unique values
        if len(series) > 0:
            unique_ratio = series.nunique() / len(series)
            return unique_ratio < 0.1 and series.nunique() < 50
        
        return False
    
    def _get_categorical_subtype(self, series: pd.Series) -> str:
        """Determine categorical subtype"""
        nunique = series.nunique()
        
        if nunique == 2:
            return 'binary'
        elif nunique <= 5:
            return 'low_cardinality'
        elif nunique <= 20:
            return 'medium_cardinality'
        else:
            return 'high_cardinality'
    
    def _get_categorical_properties(self, series: pd.Series) -> List[str]:
        """Extract categorical properties"""
        properties = []
        
        if series.nunique() == 2:
            properties.append('binary')
        
        # Check for ordinal patterns
        if self._is_ordinal(series):
            properties.append('ordinal')
        
        return properties
    
    def _is_ordinal(self, series: pd.Series) -> bool:
        """Detect ordinal categorical variables"""
        ordinal_patterns = [
            r'(low|medium|high)',
            r'(small|medium|large)',
            r'(poor|fair|good|excellent)',
            r'(never|rarely|sometimes|often|always)'
        ]
        
        values = series.dropna().astype(str).str.lower()
        for pattern in ordinal_patterns:
            if values.str.contains(pattern, regex=True).any():
                return True
        
        return False
    
    def _get_numerical_subtype(self, series: pd.Series) -> str:
        """Determine numerical subtype"""
        if pd.api.types.is_integer_dtype(series):
            if series.min() >= 0 and series.max() <= 1:
                return 'binary_numeric'
            elif series.nunique() < 10:
                return 'discrete_low'
            else:
                return 'discrete'
        else:
            return 'continuous'
    
    def _get_numerical_properties(self, series: pd.Series) -> List[str]:
        """Extract numerical properties"""
        properties = []
        
        if (series >= 0).all():
            properties.append('non_negative')
        
        if series.nunique() == 2:
            properties.append('binary')
        
        # Check for potential counts
        if pd.api.types.is_integer_dtype(series) and (series >= 0).all():
            properties.append('count_like')
        
        return properties
    
    def _is_potential_id(self, series: pd.Series, col_name: str) -> bool:
        """Detect potential ID columns"""
        # Check column name patterns
        id_patterns = ['id', 'key', 'uuid', 'identifier']
        if any(pattern in col_name.lower() for pattern in id_patterns):
            return True
        
        # Check if all values are unique
        if series.nunique() == len(series.dropna()):
            return True
        
        return False
    
    def _get_text_subtype(self, series: pd.Series) -> str:
        """Determine text subtype"""
        avg_length = series.dropna().astype(str).str.len().mean()
        
        if avg_length < 20:
            return 'short_text'
        elif avg_length < 100:
            return 'medium_text'
        else:
            return 'long_text'
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        quality = {
            'missing_data': {},
            'duplicates': len(df) - len(df.drop_duplicates()),
            'overall_completeness': 0.0,
            'quality_score': 0.0,
            'issues': []
        }
        
        # Analyze missing data
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df) * 100
            quality['missing_data'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_pct)
            }
        
        # Calculate overall completeness
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        quality['overall_completeness'] = float((total_cells - missing_cells) / total_cells * 100)
        
        # Identify quality issues
        if quality['overall_completeness'] < 80:
            quality['issues'].append('high_missing_data')
        
        if quality['duplicates'] > len(df) * 0.1:
            quality['issues'].append('many_duplicates')
        
        # Calculate quality score (0-100)
        completeness_score = quality['overall_completeness']
        duplicate_penalty = min(quality['duplicates'] / len(df) * 100, 30)
        quality['quality_score'] = max(0, completeness_score - duplicate_penalty)
        
        return quality
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect important patterns in the dataset"""
        patterns = {
            'timeseries': self._detect_timeseries_pattern(df),
            'relationships': self._detect_potential_relationships(df),
            'distributions': self._analyze_distributions(df),
            'anomalies': self._detect_potential_anomalies(df)
        }
        
        return patterns
    
    def _detect_timeseries_pattern(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if this is timeseries data"""
        datetime_cols = []
        for col in df.columns:
            if self._is_datetime_column(df[col]):
                datetime_cols.append(col)
        
        if datetime_cols:
            return {
                'is_timeseries': True,
                'datetime_columns': datetime_cols,
                'frequency': self._estimate_frequency(df, datetime_cols[0]) if datetime_cols else None
            }
        
        return {'is_timeseries': False}
    
    def _estimate_frequency(self, df: pd.DataFrame, datetime_col: str) -> str:
        """Estimate timeseries frequency"""
        try:
            dt_series = pd.to_datetime(df[datetime_col].dropna()).sort_values()
            if len(dt_series) > 1:
                diff = dt_series.diff().dropna()
                median_diff = diff.median()
                
                if median_diff.days >= 365:
                    return 'yearly'
                elif median_diff.days >= 28:
                    return 'monthly'
                elif median_diff.days >= 7:
                    return 'weekly'
                elif median_diff.days >= 1:
                    return 'daily'
                else:
                    return 'sub_daily'
        except:
            pass
        
        return 'irregular'
    
    def _detect_potential_relationships(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect potential relationships between columns"""
        relationships = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Look for potential correlations
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.5:
                        relationships.append({
                            'type': 'correlation',
                            'columns': [col1, col2],
                            'strength': abs(corr_val),
                            'direction': 'positive' if corr_val > 0 else 'negative'
                        })
        
        return relationships
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze distribution types of numerical columns"""
        distributions = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                # Simple distribution classification
                skewness = series.skew()
                if abs(skewness) < 0.5:
                    distributions[col] = 'normal'
                elif skewness > 0.5:
                    distributions[col] = 'right_skewed'
                else:
                    distributions[col] = 'left_skewed'
        
        return distributions
    
    def _detect_potential_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential anomalies that might affect analysis"""
        anomalies = {'columns_with_outliers': []}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 0:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
                
                if len(outliers) > len(series) * 0.05:  # More than 5% outliers
                    anomalies['columns_with_outliers'].append({
                        'column': col,
                        'outlier_count': len(outliers),
                        'outlier_percentage': len(outliers) / len(series) * 100
                    })
        
        return anomalies
    
    def _assess_complexity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess dataset complexity for analysis planning"""
        rows, cols = df.shape
        
        # Calculate complexity score
        size_factor = min(rows * cols / 10000, 5)  # Cap at 5
        type_diversity = len(set(str(dtype) for dtype in df.dtypes)) / 5  # Normalized to 0-1
        missing_penalty = df.isnull().sum().sum() / df.size
        
        complexity_score = size_factor + type_diversity - missing_penalty
        
        if complexity_score < 1:
            level = 'simple'
        elif complexity_score < 3:
            level = 'moderate'
        else:
            level = 'complex'
        
        return {
            'score': float(complexity_score),
            'level': level,
            'factors': {
                'size': float(size_factor),
                'type_diversity': float(type_diversity),
                'missing_data': float(missing_penalty)
            }
        }
    
    def _identify_opportunities(self, df: pd.DataFrame) -> List[str]:
        """Identify analysis opportunities based on data characteristics"""
        opportunities = []
        
        # Check for correlation analysis opportunities
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        if numeric_cols >= 2:
            opportunities.append('correlation_analysis')
        
        # Check for distribution analysis
        if numeric_cols >= 1:
            opportunities.append('distribution_analysis')
        
        # Check for categorical analysis
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        if categorical_cols >= 1:
            opportunities.append('categorical_analysis')
        
        # Check for timeseries analysis
        if self._detect_timeseries_pattern(df)['is_timeseries']:
            opportunities.append('timeseries_analysis')
        
        # Check for clustering opportunities
        if numeric_cols >= 2 and len(df) >= 50:
            opportunities.append('clustering_analysis')
        
        # Check for anomaly detection
        if numeric_cols >= 1 and len(df) >= 30:
            opportunities.append('anomaly_detection')
        
        return opportunities
    
    def _get_fallback_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback profile when main profiling fails"""
        return {
            'dataset_id': 'fallback_' + str(hash(str(df.columns)))[:8],
            'basic_stats': {
                'shape': {'rows': len(df), 'columns': len(df.columns)},
                'column_names': list(df.columns),
                'size_category': 'unknown'
            },
            'complexity': {'level': 'unknown', 'score': 0},
            'analysis_opportunities': ['basic_statistics'],
            'quality_issues': ['profiling_failed']
        }