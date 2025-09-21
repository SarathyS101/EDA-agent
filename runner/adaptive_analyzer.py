#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.stats import chi2_contingency
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import asyncio

from agentic_planner import AnalysisStrategy
from webhook_system import AgenticWebhookManager, WebhookEvent

logger = logging.getLogger(__name__)

class AnalysisQuality(Enum):
    """Quality levels for analysis results"""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"

class AdaptiveAnalyzer:
    """
    Adaptive analysis toolkit that can execute different strategies based on
    the planner's decisions and adapt its approach based on intermediate results.
    """
    
    def __init__(self, webhook_manager: AgenticWebhookManager = None):
        self.webhook_manager = webhook_manager
        self.analysis_results = {}
        self.strategy_implementations = {}
        self.quality_evaluators = {}
        self.fallback_strategies = {}
        
        # Configure plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Initialize strategy implementations
        self._initialize_strategy_implementations()
        self._initialize_quality_evaluators()
        self._initialize_fallback_strategies()
    
    def _initialize_strategy_implementations(self):
        """Initialize implementations for each analysis strategy"""
        self.strategy_implementations = {
            AnalysisStrategy.BASIC_STATS: self._execute_basic_stats,
            AnalysisStrategy.CORRELATION_MATRIX: self._execute_correlation_analysis,
            AnalysisStrategy.DISTRIBUTION_ANALYSIS: self._execute_distribution_analysis,
            AnalysisStrategy.CLUSTERING: self._execute_clustering_analysis,
            AnalysisStrategy.DIMENSIONALITY_REDUCTION: self._execute_dimensionality_reduction,
            AnalysisStrategy.ANOMALY_DETECTION: self._execute_anomaly_detection,
            AnalysisStrategy.TIME_SERIES: self._execute_time_series_analysis,
            AnalysisStrategy.CATEGORICAL_ANALYSIS: self._execute_categorical_analysis,
            AnalysisStrategy.TEXT_ANALYSIS: self._execute_text_analysis,
            AnalysisStrategy.PREDICTIVE_MODELING: self._execute_predictive_modeling
        }
    
    def _initialize_quality_evaluators(self):
        """Initialize quality evaluation functions for each strategy"""
        self.quality_evaluators = {
            AnalysisStrategy.BASIC_STATS: self._evaluate_basic_stats_quality,
            AnalysisStrategy.CORRELATION_MATRIX: self._evaluate_correlation_quality,
            AnalysisStrategy.DISTRIBUTION_ANALYSIS: self._evaluate_distribution_quality,
            AnalysisStrategy.CLUSTERING: self._evaluate_clustering_quality,
            AnalysisStrategy.ANOMALY_DETECTION: self._evaluate_anomaly_quality,
            AnalysisStrategy.TIME_SERIES: self._evaluate_timeseries_quality,
            AnalysisStrategy.CATEGORICAL_ANALYSIS: self._evaluate_categorical_quality
        }
    
    def _initialize_fallback_strategies(self):
        """Initialize fallback strategies for when primary analysis fails"""
        self.fallback_strategies = {
            AnalysisStrategy.CORRELATION_MATRIX: [
                {'method': 'spearman', 'reason': 'non_parametric_alternative'},
                {'method': 'kendall', 'reason': 'rank_based_correlation'},
                {'method': 'mutual_information', 'reason': 'non_linear_relationships'}
            ],
            AnalysisStrategy.CLUSTERING: [
                {'method': 'dbscan', 'reason': 'density_based_clustering'},
                {'method': 'gaussian_mixture', 'reason': 'soft_clustering'},
                {'method': 'hierarchical', 'reason': 'agglomerative_approach'}
            ],
            AnalysisStrategy.ANOMALY_DETECTION: [
                {'method': 'local_outlier_factor', 'reason': 'local_density_based'},
                {'method': 'one_class_svm', 'reason': 'support_vector_approach'},
                {'method': 'statistical_outliers', 'reason': 'simple_statistical_method'}
            ]
        }
    
    async def execute_strategy(self, strategy: AnalysisStrategy, df: pd.DataFrame, 
                             strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a specific analysis strategy with adaptive behavior.
        This method demonstrates autonomous decision-making and quality monitoring.
        """
        logger.info(f"Executing strategy: {strategy.value}")
        
        # Notify webhook system about strategy execution
        if self.webhook_manager:
            await self.webhook_manager.notify_phase_complete(
                f"{strategy.value}_started", 
                {'strategy': strategy.value, 'config': strategy_config}
            )
        
        try:
            # Execute primary strategy
            result = await self._execute_primary_strategy(strategy, df, strategy_config)
            
            # Evaluate quality
            quality = self._evaluate_result_quality(strategy, result, df)
            
            # Apply adaptive behavior based on quality
            if quality in [AnalysisQuality.POOR, AnalysisQuality.FAILED]:
                result = await self._apply_fallback_strategy(strategy, df, strategy_config, result)
                quality = self._evaluate_result_quality(strategy, result, df)
            
            # Add metadata
            result['quality'] = quality.value
            result['strategy'] = strategy.value
            result['adaptive_applied'] = result.get('fallback_used', False)
            
            # Store result
            self.analysis_results[strategy.value] = result
            
            # Notify webhook system about completion
            if self.webhook_manager:
                await self.webhook_manager.notify_phase_complete(
                    strategy.value, result
                )
                
                # Trigger quality check if needed
                if quality in [AnalysisQuality.POOR, AnalysisQuality.FAILED]:
                    await self.webhook_manager.notify_quality_issue(
                        'low_quality_results',
                        {'strategy': strategy.value, 'quality': quality.value}
                    )
            
            logger.info(f"Strategy {strategy.value} completed with quality: {quality.value}")
            return result
            
        except Exception as e:
            logger.error(f"Strategy {strategy.value} failed: {str(e)}")
            
            # Notify webhook system about error
            if self.webhook_manager:
                await self.webhook_manager.webhook_trigger.trigger_event(
                    WebhookEvent.ERROR_OCCURRED,
                    {
                        'strategy': strategy.value,
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
                )
            
            # Return error result
            return {
                'strategy': strategy.value,
                'quality': AnalysisQuality.FAILED.value,
                'error': str(e),
                'success': False
            }
    
    async def _execute_primary_strategy(self, strategy: AnalysisStrategy, df: pd.DataFrame,
                                      config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the primary implementation of a strategy"""
        implementation = self.strategy_implementations.get(strategy)
        if not implementation:
            raise ValueError(f"No implementation found for strategy: {strategy}")
        
        return await implementation(df, config)
    
    def _evaluate_result_quality(self, strategy: AnalysisStrategy, result: Dict[str, Any],
                               df: pd.DataFrame) -> AnalysisQuality:
        """Evaluate the quality of analysis results"""
        evaluator = self.quality_evaluators.get(strategy)
        if not evaluator:
            return AnalysisQuality.FAIR  # Default quality
        
        return evaluator(result, df)
    
    async def _apply_fallback_strategy(self, strategy: AnalysisStrategy, df: pd.DataFrame,
                                     original_config: Dict[str, Any], 
                                     failed_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fallback strategies when primary strategy fails or produces poor results"""
        fallbacks = self.fallback_strategies.get(strategy, [])
        
        for fallback in fallbacks:
            try:
                logger.info(f"Applying fallback: {fallback['method']} - {fallback['reason']}")
                
                # Modify config for fallback
                fallback_config = original_config.copy()
                fallback_config.update(fallback)
                
                # Execute fallback
                result = await self._execute_primary_strategy(strategy, df, fallback_config)
                result['fallback_used'] = True
                result['fallback_method'] = fallback['method']
                result['fallback_reason'] = fallback['reason']
                
                # Check if fallback improved quality
                quality = self._evaluate_result_quality(strategy, result, df)
                if quality not in [AnalysisQuality.POOR, AnalysisQuality.FAILED]:
                    logger.info(f"Fallback {fallback['method']} succeeded")
                    return result
                
            except Exception as e:
                logger.warning(f"Fallback {fallback['method']} failed: {str(e)}")
                continue
        
        # If all fallbacks failed, return the original result with fallback info
        failed_result['fallback_attempted'] = True
        failed_result['all_fallbacks_failed'] = True
        return failed_result
    
    # Strategy Implementations
    
    async def _execute_basic_stats(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute basic statistics analysis"""
        try:
            result = {
                'success': True,
                'summary_statistics': {},
                'data_info': {},
                'missing_data': {},
                'data_types': {}
            }
            
            # Basic descriptive statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                result['summary_statistics'] = df[numeric_cols].describe().to_dict()
            
            # Data info
            result['data_info'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum()
            }
            
            # Missing data analysis
            missing_data = df.isnull().sum()
            result['missing_data'] = {
                'counts': missing_data.to_dict(),
                'percentages': (missing_data / len(df) * 100).to_dict(),
                'total_missing': missing_data.sum(),
                'completeness': (1 - missing_data.sum() / df.size) * 100
            }
            
            # Data types
            result['data_types'] = df.dtypes.astype(str).to_dict()
            
            return result
            
        except Exception as e:
            logger.error(f"Basic stats analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_correlation_analysis(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute correlation analysis with adaptive method selection"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {
                    'success': False,
                    'error': 'Insufficient numeric columns for correlation analysis',
                    'numeric_columns_found': len(numeric_cols)
                }
            
            method = config.get('method', 'pearson')
            threshold = config.get('threshold', 0.5)
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr(method=method)
            
            # Find strong correlations
            strong_correlations = []
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols[i+1:], i+1):
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) >= threshold:
                        strong_correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': float(corr_val),
                            'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                        })
            
            # Generate visualization
            viz_path = await self._create_correlation_heatmap(corr_matrix)
            
            return {
                'success': True,
                'correlation_matrix': corr_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'method_used': method,
                'threshold_used': threshold,
                'total_correlations_found': len(strong_correlations),
                'visualization_path': viz_path
            }
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_distribution_analysis(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute distribution analysis with normality testing"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return {
                    'success': False,
                    'error': 'No numeric columns found for distribution analysis'
                }
            
            distributions = {}
            outliers = {}
            normality_tests = {}
            
            for col in numeric_cols:
                series = df[col].dropna()
                
                if len(series) == 0:
                    continue
                
                # Distribution characteristics
                distributions[col] = {
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std()),
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis()),
                    'min': float(series.min()),
                    'max': float(series.max())
                }
                
                # Outlier detection using IQR method
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)
                outliers[col] = {
                    'count': int(outlier_mask.sum()),
                    'percentage': float(outlier_mask.sum() / len(series) * 100),
                    'values': series[outlier_mask].tolist()[:10]  # Limit to first 10
                }
                
                # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for large)
                if len(series) <= 5000:
                    stat, p_value = stats.shapiro(series)
                    test_name = 'shapiro_wilk'
                else:
                    # Use Anderson-Darling test for large samples
                    result = stats.anderson(series)
                    stat, p_value = result.statistic, 0.05  # Approximation
                    test_name = 'anderson_darling'
                
                normality_tests[col] = {
                    'test': test_name,
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
            
            # Generate visualizations
            viz_paths = await self._create_distribution_plots(df, numeric_cols)
            
            return {
                'success': True,
                'distributions': distributions,
                'outliers': outliers,
                'normality_tests': normality_tests,
                'visualization_paths': viz_paths
            }
            
        except Exception as e:
            logger.error(f"Distribution analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_clustering_analysis(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute clustering analysis with adaptive algorithm selection"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {
                    'success': False,
                    'error': 'Insufficient numeric columns for clustering analysis'
                }
            
            # Prepare data
            X = df[numeric_cols].dropna()
            
            if len(X) < 10:
                return {
                    'success': False,
                    'error': 'Insufficient data points for clustering'
                }
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            algorithm = config.get('algorithm', 'kmeans')
            k_range = config.get('k_range', [2, 8])
            
            if algorithm == 'kmeans':
                best_k, best_score, cluster_labels = await self._find_optimal_kmeans(X_scaled, k_range)
            elif algorithm == 'dbscan':
                cluster_labels, best_score = await self._execute_dbscan(X_scaled)
                best_k = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            elif algorithm == 'hierarchical':
                best_k = config.get('n_clusters', 3)
                cluster_labels, best_score = await self._execute_hierarchical(X_scaled, best_k)
            else:
                raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
            
            # Analyze clusters
            cluster_analysis = self._analyze_clusters(X, cluster_labels, numeric_cols)
            
            # Generate visualization
            viz_path = await self._create_cluster_plot(X_scaled, cluster_labels)
            
            return {
                'success': True,
                'algorithm_used': algorithm,
                'optimal_clusters': int(best_k),
                'silhouette_score': float(best_score),
                'cluster_labels': cluster_labels.tolist(),
                'cluster_analysis': cluster_analysis,
                'visualization_path': viz_path
            }
            
        except Exception as e:
            logger.error(f"Clustering analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_dimensionality_reduction(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dimensionality reduction (PCA)"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 3:
                return {
                    'success': False,
                    'error': 'Need at least 3 numeric columns for dimensionality reduction'
                }
            
            X = df[numeric_cols].dropna()
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Perform PCA
            n_components = config.get('n_components', min(len(numeric_cols), 5))
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Feature importance
            feature_importance = {}
            for i, component in enumerate(pca.components_):
                feature_importance[f'PC{i+1}'] = {
                    col: float(abs(weight)) 
                    for col, weight in zip(numeric_cols, component)
                }
            
            return {
                'success': True,
                'n_components': n_components,
                'explained_variance_ratio': explained_variance.tolist(),
                'cumulative_variance_ratio': cumulative_variance.tolist(),
                'total_variance_explained': float(cumulative_variance[-1]),
                'feature_importance': feature_importance,
                'transformed_data': X_pca.tolist()[:100]  # Limit for JSON size
            }
            
        except Exception as e:
            logger.error(f"Dimensionality reduction failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_anomaly_detection(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute anomaly detection with multiple methods"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return {
                    'success': False,
                    'error': 'No numeric columns found for anomaly detection'
                }
            
            X = df[numeric_cols].dropna()
            contamination = config.get('contamination', 0.1)
            
            # Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_scores = iso_forest.decision_function(X)
            anomaly_labels = iso_forest.predict(X)
            
            # Statistical outliers (IQR method)
            statistical_outliers = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)].index
                statistical_outliers.extend(outliers)
            
            statistical_outliers = list(set(statistical_outliers))
            
            # Combine results
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'success': True,
                'isolation_forest_anomalies': anomaly_indices.tolist(),
                'statistical_outliers': statistical_outliers,
                'anomaly_scores': anomaly_scores.tolist(),
                'total_anomalies': len(anomaly_indices),
                'contamination_rate': float(len(anomaly_indices) / len(X)),
                'method': 'isolation_forest_with_statistical'
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_time_series_analysis(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute time series analysis"""
        try:
            # Find datetime columns
            datetime_cols = []
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
                elif pd.api.types.is_string_dtype(df[col]):
                    try:
                        pd.to_datetime(df[col].head(10))
                        datetime_cols.append(col)
                    except:
                        continue
            
            if not datetime_cols:
                return {
                    'success': False,
                    'error': 'No datetime columns found for time series analysis'
                }
            
            time_col = datetime_cols[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return {
                    'success': False,
                    'error': 'No numeric columns found for time series analysis'
                }
            
            # Prepare time series data
            df_ts = df.copy()
            df_ts[time_col] = pd.to_datetime(df_ts[time_col])
            df_ts = df_ts.sort_values(time_col).set_index(time_col)
            
            # Analyze each numeric column
            results = {}
            for col in numeric_cols:
                series = df_ts[col].dropna()
                if len(series) < 10:
                    continue
                
                # Basic trend analysis
                x = np.arange(len(series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
                
                results[col] = {
                    'trend_slope': float(slope),
                    'trend_r_squared': float(r_value**2),
                    'trend_p_value': float(p_value),
                    'has_trend': p_value < 0.05,
                    'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                }
            
            return {
                'success': True,
                'datetime_column': time_col,
                'time_range': {
                    'start': df_ts.index.min().isoformat(),
                    'end': df_ts.index.max().isoformat()
                },
                'series_analysis': results
            }
            
        except Exception as e:
            logger.error(f"Time series analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_categorical_analysis(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute categorical data analysis"""
        try:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            if len(categorical_cols) == 0:
                return {
                    'success': False,
                    'error': 'No categorical columns found'
                }
            
            results = {}
            
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                
                results[col] = {
                    'unique_values': int(df[col].nunique()),
                    'most_frequent': str(value_counts.index[0]),
                    'most_frequent_count': int(value_counts.iloc[0]),
                    'least_frequent': str(value_counts.index[-1]),
                    'least_frequent_count': int(value_counts.iloc[-1]),
                    'value_distribution': value_counts.head(10).to_dict(),
                    'missing_values': int(df[col].isnull().sum())
                }
            
            # Cross-tabulation analysis for pairs of categorical variables
            associations = []
            if len(categorical_cols) >= 2:
                for i, col1 in enumerate(categorical_cols[:5]):  # Limit to avoid explosion
                    for col2 in categorical_cols[i+1:6]:
                        try:
                            contingency_table = pd.crosstab(df[col1], df[col2])
                            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                            
                            associations.append({
                                'column1': col1,
                                'column2': col2,
                                'chi2_statistic': float(chi2),
                                'p_value': float(p_value),
                                'is_associated': p_value < 0.05,
                                'degrees_of_freedom': int(dof)
                            })
                        except Exception as e:
                            logger.warning(f"Chi-square test failed for {col1} vs {col2}: {str(e)}")
                            continue
            
            return {
                'success': True,
                'categorical_analysis': results,
                'associations': associations,
                'total_categorical_columns': len(categorical_cols)
            }
            
        except Exception as e:
            logger.error(f"Categorical analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_text_analysis(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute basic text analysis"""
        try:
            text_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if it's likely text (average length > 10)
                    avg_length = df[col].astype(str).str.len().mean()
                    if avg_length > 10:
                        text_cols.append(col)
            
            if not text_cols:
                return {
                    'success': False,
                    'error': 'No text columns identified'
                }
            
            results = {}
            for col in text_cols:
                text_series = df[col].dropna().astype(str)
                
                results[col] = {
                    'total_texts': len(text_series),
                    'avg_length': float(text_series.str.len().mean()),
                    'max_length': int(text_series.str.len().max()),
                    'min_length': int(text_series.str.len().min()),
                    'unique_texts': int(text_series.nunique()),
                    'most_common_words': self._get_common_words(text_series)
                }
            
            return {
                'success': True,
                'text_analysis': results,
                'text_columns_found': len(text_cols)
            }
            
        except Exception as e:
            logger.error(f"Text analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_predictive_modeling(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute basic predictive modeling assessment"""
        # This is a placeholder for more complex predictive modeling
        return {
            'success': False,
            'error': 'Predictive modeling not implemented in this version'
        }
    
    # Helper methods for clustering
    
    async def _find_optimal_kmeans(self, X: np.ndarray, k_range: List[int]) -> Tuple[int, float, np.ndarray]:
        """Find optimal K for K-means clustering"""
        best_k = k_range[0]
        best_score = -1
        best_labels = None
        
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            if len(set(labels)) > 1:  # Need at least 2 clusters for silhouette score
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
        
        return best_k, best_score, best_labels
    
    async def _execute_dbscan(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Execute DBSCAN clustering"""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
        else:
            score = 0.0
        
        return labels, score
    
    async def _execute_hierarchical(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, float]:
        """Execute hierarchical clustering"""
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        labels = hierarchical.fit_predict(X)
        
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
        else:
            score = 0.0
        
        return labels, score
    
    def _analyze_clusters(self, X: pd.DataFrame, labels: np.ndarray, 
                         columns: List[str]) -> Dict[str, Any]:
        """Analyze cluster characteristics"""
        cluster_analysis = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise points in DBSCAN
                continue
            
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'size': int(cluster_mask.sum()),
                'percentage': float(cluster_mask.sum() / len(labels) * 100),
                'centroids': cluster_data[columns].mean().to_dict()
            }
        
        return cluster_analysis
    
    def _get_common_words(self, text_series: pd.Series, top_n: int = 10) -> List[str]:
        """Get most common words from text series"""
        try:
            # Simple word extraction (split by spaces and common punctuation)
            all_text = ' '.join(text_series.values)
            words = all_text.lower().split()
            
            # Remove common stop words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = [word.strip('.,!?;:"()[]{}') for word in words 
                    if len(word) > 2 and word.lower() not in stop_words]
            
            # Count words
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Return top N
            return sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        except Exception:
            return []
    
    # Visualization methods
    
    async def _create_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> str:
        """Create correlation heatmap"""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
            
            viz_path = '/tmp/correlation_heatmap.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return viz_path
        except Exception as e:
            logger.error(f"Failed to create correlation heatmap: {str(e)}")
            return None
    
    async def _create_distribution_plots(self, df: pd.DataFrame, columns: List[str]) -> List[str]:
        """Create distribution plots"""
        viz_paths = []
        
        try:
            for col in columns[:6]:  # Limit to 6 plots
                plt.figure(figsize=(10, 6))
                
                # Histogram with KDE
                plt.subplot(1, 2, 1)
                df[col].hist(bins=30, alpha=0.7, edgecolor='black')
                plt.title(f'Histogram: {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                
                # Box plot
                plt.subplot(1, 2, 2)
                df[col].plot.box()
                plt.title(f'Box Plot: {col}')
                plt.ylabel(col)
                
                plt.tight_layout()
                
                viz_path = f'/tmp/distribution_{col}.png'
                plt.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                viz_paths.append(viz_path)
                
        except Exception as e:
            logger.error(f"Failed to create distribution plots: {str(e)}")
        
        return viz_paths
    
    async def _create_cluster_plot(self, X: np.ndarray, labels: np.ndarray) -> str:
        """Create cluster visualization"""
        try:
            # Use PCA for 2D visualization if more than 2 dimensions
            if X.shape[1] > 2:
                pca = PCA(n_components=2)
                X_viz = pca.fit_transform(X)
            else:
                X_viz = X
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_viz[:, 0], X_viz[:, 1], c=labels, cmap='viridis')
            plt.colorbar(scatter)
            plt.title('Cluster Visualization')
            plt.xlabel('Component 1' if X.shape[1] > 2 else 'Feature 1')
            plt.ylabel('Component 2' if X.shape[1] > 2 else 'Feature 2')
            
            viz_path = '/tmp/cluster_plot.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return viz_path
            
        except Exception as e:
            logger.error(f"Failed to create cluster plot: {str(e)}")
            return None
    
    # Quality evaluation methods
    
    def _evaluate_basic_stats_quality(self, result: Dict[str, Any], df: pd.DataFrame) -> AnalysisQuality:
        """Evaluate quality of basic statistics results"""
        if not result.get('success', False):
            return AnalysisQuality.FAILED
        
        completeness = result.get('missing_data', {}).get('completeness', 0)
        
        if completeness >= 95:
            return AnalysisQuality.EXCELLENT
        elif completeness >= 80:
            return AnalysisQuality.GOOD
        elif completeness >= 60:
            return AnalysisQuality.FAIR
        else:
            return AnalysisQuality.POOR
    
    def _evaluate_correlation_quality(self, result: Dict[str, Any], df: pd.DataFrame) -> AnalysisQuality:
        """Evaluate quality of correlation analysis results"""
        if not result.get('success', False):
            return AnalysisQuality.FAILED
        
        strong_correlations = len(result.get('strong_correlations', []))
        total_possible = len(df.select_dtypes(include=[np.number]).columns)
        
        if total_possible < 2:
            return AnalysisQuality.POOR
        
        if strong_correlations > 0:
            return AnalysisQuality.GOOD
        else:
            return AnalysisQuality.FAIR  # No strong correlations found, but analysis succeeded
    
    def _evaluate_distribution_quality(self, result: Dict[str, Any], df: pd.DataFrame) -> AnalysisQuality:
        """Evaluate quality of distribution analysis results"""
        if not result.get('success', False):
            return AnalysisQuality.FAILED
        
        distributions = result.get('distributions', {})
        if not distributions:
            return AnalysisQuality.POOR
        
        return AnalysisQuality.GOOD
    
    def _evaluate_clustering_quality(self, result: Dict[str, Any], df: pd.DataFrame) -> AnalysisQuality:
        """Evaluate quality of clustering results"""
        if not result.get('success', False):
            return AnalysisQuality.FAILED
        
        silhouette_score = result.get('silhouette_score', 0)
        
        if silhouette_score >= 0.7:
            return AnalysisQuality.EXCELLENT
        elif silhouette_score >= 0.5:
            return AnalysisQuality.GOOD
        elif silhouette_score >= 0.2:
            return AnalysisQuality.FAIR
        else:
            return AnalysisQuality.POOR
    
    def _evaluate_anomaly_quality(self, result: Dict[str, Any], df: pd.DataFrame) -> AnalysisQuality:
        """Evaluate quality of anomaly detection results"""
        if not result.get('success', False):
            return AnalysisQuality.FAILED
        
        contamination_rate = result.get('contamination_rate', 0)
        
        # Quality based on reasonable contamination rate
        if 0.01 <= contamination_rate <= 0.15:
            return AnalysisQuality.GOOD
        elif contamination_rate <= 0.25:
            return AnalysisQuality.FAIR
        else:
            return AnalysisQuality.POOR
    
    def _evaluate_timeseries_quality(self, result: Dict[str, Any], df: pd.DataFrame) -> AnalysisQuality:
        """Evaluate quality of time series analysis results"""
        if not result.get('success', False):
            return AnalysisQuality.FAILED
        
        series_analysis = result.get('series_analysis', {})
        if not series_analysis:
            return AnalysisQuality.POOR
        
        # Check if trends were detected
        trends_detected = sum(1 for analysis in series_analysis.values() 
                            if analysis.get('has_trend', False))
        
        if trends_detected > 0:
            return AnalysisQuality.GOOD
        else:
            return AnalysisQuality.FAIR
    
    def _evaluate_categorical_quality(self, result: Dict[str, Any], df: pd.DataFrame) -> AnalysisQuality:
        """Evaluate quality of categorical analysis results"""
        if not result.get('success', False):
            return AnalysisQuality.FAILED
        
        categorical_analysis = result.get('categorical_analysis', {})
        associations = result.get('associations', [])
        
        if not categorical_analysis:
            return AnalysisQuality.POOR
        
        # Quality based on number of associations found
        significant_associations = sum(1 for assoc in associations 
                                     if assoc.get('is_associated', False))
        
        if significant_associations > 0:
            return AnalysisQuality.GOOD
        else:
            return AnalysisQuality.FAIR