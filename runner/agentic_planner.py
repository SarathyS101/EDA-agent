#!/usr/bin/env python3

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import asyncio
from dataclasses import dataclass

from data_profiler import DataProfiler
from webhook_system import AgenticWebhookManager, WebhookEvent

logger = logging.getLogger(__name__)

class AnalysisGoal(Enum):
    """High-level analysis goals that guide the agent's decision-making"""
    EXPLORE_PATTERNS = "explore_patterns"
    FIND_CORRELATIONS = "find_correlations"
    DETECT_ANOMALIES = "detect_anomalies"
    UNDERSTAND_DISTRIBUTIONS = "understand_distributions"
    PREDICT_RELATIONSHIPS = "predict_relationships"
    SEGMENT_DATA = "segment_data"
    TIME_ANALYSIS = "time_analysis"
    QUALITY_ASSESSMENT = "quality_assessment"

class AnalysisStrategy(Enum):
    """Specific analysis strategies the agent can choose"""
    BASIC_STATS = "basic_statistics"
    CORRELATION_MATRIX = "correlation_analysis"
    DISTRIBUTION_ANALYSIS = "distribution_analysis"
    CLUSTERING = "clustering_analysis"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series_analysis"
    CATEGORICAL_ANALYSIS = "categorical_analysis"
    TEXT_ANALYSIS = "text_analysis"
    PREDICTIVE_MODELING = "predictive_modeling"

@dataclass
class AnalysisTask:
    """Individual analysis task with priority and dependencies"""
    strategy: AnalysisStrategy
    priority: int  # 1-10, 10 being highest
    dependencies: List[AnalysisStrategy]
    estimated_time: int  # seconds
    confidence_threshold: float  # minimum confidence to proceed
    metadata: Dict[str, Any]
    
    def can_execute(self, completed_strategies: List[AnalysisStrategy]) -> bool:
        """Check if this task can be executed based on dependencies"""
        return all(dep in completed_strategies for dep in self.dependencies)

class AgenticAnalysisPlanner:
    """
    The core planning engine that makes the system truly agentic.
    It analyzes data characteristics and dynamically chooses analysis strategies,
    adapts based on results, and makes autonomous decisions about next steps.
    """
    
    def __init__(self, profiler: DataProfiler = None):
        self.profiler = profiler or DataProfiler()
        self.goals = set()
        self.strategy_registry = {}
        self.execution_history = []
        self.current_plan = None
        self.adaptation_rules = {}
        self.quality_thresholds = {
            'min_data_quality': 0.6,
            'min_result_confidence': 0.7,
            'max_missing_data_pct': 30,
            'min_variance_threshold': 0.01
        }
        
        self._initialize_strategy_registry()
        self._initialize_adaptation_rules()
    
    def _initialize_strategy_registry(self):
        """Initialize the registry of available analysis strategies"""
        self.strategy_registry = {
            AnalysisStrategy.BASIC_STATS: {
                'function': self._plan_basic_stats,
                'prerequisites': ['data_loaded'],
                'outputs': ['summary_stats', 'data_types'],
                'time_complexity': 'O(n)',
                'memory_complexity': 'O(1)'
            },
            AnalysisStrategy.CORRELATION_MATRIX: {
                'function': self._plan_correlation_analysis,
                'prerequisites': ['numeric_columns'],
                'outputs': ['correlation_matrix', 'strong_correlations'],
                'time_complexity': 'O(n²)',
                'memory_complexity': 'O(n²)'
            },
            AnalysisStrategy.DISTRIBUTION_ANALYSIS: {
                'function': self._plan_distribution_analysis,
                'prerequisites': ['numeric_columns'],
                'outputs': ['distribution_shapes', 'outliers'],
                'time_complexity': 'O(n log n)',
                'memory_complexity': 'O(n)'
            },
            AnalysisStrategy.CLUSTERING: {
                'function': self._plan_clustering_analysis,
                'prerequisites': ['numeric_columns', 'scaled_data'],
                'outputs': ['clusters', 'cluster_centers'],
                'time_complexity': 'O(n² log n)',
                'memory_complexity': 'O(n)'
            },
            AnalysisStrategy.ANOMALY_DETECTION: {
                'function': self._plan_anomaly_detection,
                'prerequisites': ['numeric_columns'],
                'outputs': ['anomalies', 'anomaly_scores'],
                'time_complexity': 'O(n log n)',
                'memory_complexity': 'O(n)'
            },
            AnalysisStrategy.TIME_SERIES: {
                'function': self._plan_time_series_analysis,
                'prerequisites': ['datetime_column', 'ordered_data'],
                'outputs': ['trends', 'seasonality', 'forecasts'],
                'time_complexity': 'O(n log n)',
                'memory_complexity': 'O(n)'
            },
            AnalysisStrategy.CATEGORICAL_ANALYSIS: {
                'function': self._plan_categorical_analysis,
                'prerequisites': ['categorical_columns'],
                'outputs': ['category_distributions', 'associations'],
                'time_complexity': 'O(n)',
                'memory_complexity': 'O(k)'  # k = unique categories
            }
        }
    
    def _initialize_adaptation_rules(self):
        """Initialize rules for adapting analysis based on intermediate results"""
        self.adaptation_rules = {
            'low_correlation_found': {
                'condition': lambda results: max(results.get('correlations', [0])) < 0.3,
                'action': 'add_clustering_analysis',
                'reason': 'Low correlations suggest potential clusters'
            },
            'high_missing_data': {
                'condition': lambda profile: profile['data_quality']['overall_completeness'] < 70,
                'action': 'add_imputation_strategy',
                'reason': 'High missing data requires preprocessing'
            },
            'strong_temporal_pattern': {
                'condition': lambda profile: profile['patterns']['timeseries']['is_timeseries'],
                'action': 'prioritize_time_series_analysis',
                'reason': 'Temporal patterns detected'
            },
            'outliers_detected': {
                'condition': lambda results: len(results.get('outliers', [])) > len(results.get('data', [])) * 0.05,
                'action': 'add_robust_analysis',
                'reason': 'Significant outliers detected'
            },
            'high_dimensionality': {
                'condition': lambda profile: profile['basic_stats']['shape']['columns'] > 20,
                'action': 'add_dimensionality_reduction',
                'reason': 'High dimensional data benefits from reduction'
            }
        }
    
    async def create_analysis_plan(self, data_profile: Dict[str, Any], 
                                 user_goals: List[str] = None,
                                 constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a dynamic, goal-oriented analysis plan based on data characteristics.
        This is where the agent demonstrates true autonomy by reasoning about what to do.
        """
        logger.info("Creating agentic analysis plan...")
        
        # 1. Set analysis goals based on data characteristics
        self.goals = self._determine_analysis_goals(data_profile, user_goals)
        
        # 2. Generate initial task list
        initial_tasks = self._generate_initial_tasks(data_profile)
        
        # 3. Apply adaptation rules
        adapted_tasks = self._apply_adaptation_rules(initial_tasks, data_profile)
        
        # 4. Prioritize and order tasks
        ordered_tasks = self._prioritize_and_order_tasks(adapted_tasks, constraints)
        
        # 5. Create execution plan with decision points
        execution_plan = self._create_execution_plan(ordered_tasks, data_profile)
        
        # 6. Add quality gates and recovery strategies
        execution_plan = self._add_quality_gates(execution_plan)
        
        self.current_plan = execution_plan
        
        logger.info(f"Created analysis plan with {len(execution_plan['tasks'])} tasks, "
                   f"targeting goals: {[goal.value for goal in self.goals]}")
        
        return execution_plan
    
    def _determine_analysis_goals(self, data_profile: Dict[str, Any], 
                                user_goals: List[str] = None) -> set:
        """
        Autonomously determine what the analysis should achieve based on data characteristics.
        This demonstrates the agent's ability to set its own objectives.
        """
        goals = set()
        
        # Always start with basic understanding
        goals.add(AnalysisGoal.QUALITY_ASSESSMENT)
        
        # Goal determination based on data characteristics
        opportunities = data_profile.get('analysis_opportunities', [])
        
        if 'correlation_analysis' in opportunities:
            goals.add(AnalysisGoal.FIND_CORRELATIONS)
        
        if 'distribution_analysis' in opportunities:
            goals.add(AnalysisGoal.UNDERSTAND_DISTRIBUTIONS)
        
        if 'timeseries_analysis' in opportunities:
            goals.add(AnalysisGoal.TIME_ANALYSIS)
        
        if 'clustering_analysis' in opportunities:
            goals.add(AnalysisGoal.SEGMENT_DATA)
        
        if 'anomaly_detection' in opportunities:
            goals.add(AnalysisGoal.DETECT_ANOMALIES)
        
        # Add pattern exploration for complex datasets
        if data_profile.get('complexity', {}).get('level') in ['moderate', 'complex']:
            goals.add(AnalysisGoal.EXPLORE_PATTERNS)
        
        # Consider user-specified goals
        if user_goals:
            goal_mapping = {
                'correlations': AnalysisGoal.FIND_CORRELATIONS,
                'patterns': AnalysisGoal.EXPLORE_PATTERNS,
                'anomalies': AnalysisGoal.DETECT_ANOMALIES,
                'distributions': AnalysisGoal.UNDERSTAND_DISTRIBUTIONS,
                'clustering': AnalysisGoal.SEGMENT_DATA,
                'time_series': AnalysisGoal.TIME_ANALYSIS,
                'predictions': AnalysisGoal.PREDICT_RELATIONSHIPS
            }
            
            for user_goal in user_goals:
                if user_goal in goal_mapping:
                    goals.add(goal_mapping[user_goal])
        
        return goals
    
    def _generate_initial_tasks(self, data_profile: Dict[str, Any]) -> List[AnalysisTask]:
        """Generate initial analysis tasks based on goals and data characteristics"""
        tasks = []
        
        # Always start with basic statistics
        tasks.append(AnalysisTask(
            strategy=AnalysisStrategy.BASIC_STATS,
            priority=10,  # Highest priority
            dependencies=[],
            estimated_time=30,
            confidence_threshold=0.9,
            metadata={'required': True}
        ))
        
        # Add tasks based on goals
        if AnalysisGoal.FIND_CORRELATIONS in self.goals:
            tasks.append(AnalysisTask(
                strategy=AnalysisStrategy.CORRELATION_MATRIX,
                priority=8,
                dependencies=[AnalysisStrategy.BASIC_STATS],
                estimated_time=60,
                confidence_threshold=0.7,
                metadata={'goal': 'correlation_analysis'}
            ))
        
        if AnalysisGoal.UNDERSTAND_DISTRIBUTIONS in self.goals:
            tasks.append(AnalysisTask(
                strategy=AnalysisStrategy.DISTRIBUTION_ANALYSIS,
                priority=7,
                dependencies=[AnalysisStrategy.BASIC_STATS],
                estimated_time=90,
                confidence_threshold=0.8,
                metadata={'goal': 'distribution_analysis'}
            ))
        
        if AnalysisGoal.SEGMENT_DATA in self.goals:
            tasks.append(AnalysisTask(
                strategy=AnalysisStrategy.CLUSTERING,
                priority=6,
                dependencies=[AnalysisStrategy.BASIC_STATS, AnalysisStrategy.CORRELATION_MATRIX],
                estimated_time=120,
                confidence_threshold=0.6,
                metadata={'goal': 'clustering'}
            ))
        
        if AnalysisGoal.DETECT_ANOMALIES in self.goals:
            tasks.append(AnalysisTask(
                strategy=AnalysisStrategy.ANOMALY_DETECTION,
                priority=7,
                dependencies=[AnalysisStrategy.DISTRIBUTION_ANALYSIS],
                estimated_time=90,
                confidence_threshold=0.7,
                metadata={'goal': 'anomaly_detection'}
            ))
        
        if AnalysisGoal.TIME_ANALYSIS in self.goals:
            tasks.append(AnalysisTask(
                strategy=AnalysisStrategy.TIME_SERIES,
                priority=8,
                dependencies=[AnalysisStrategy.BASIC_STATS],
                estimated_time=150,
                confidence_threshold=0.8,
                metadata={'goal': 'time_series'}
            ))
        
        # Add categorical analysis if categorical data is present
        categorical_cols = data_profile.get('column_types', {}).get('categorical', [])
        if categorical_cols:
            tasks.append(AnalysisTask(
                strategy=AnalysisStrategy.CATEGORICAL_ANALYSIS,
                priority=6,
                dependencies=[AnalysisStrategy.BASIC_STATS],
                estimated_time=60,
                confidence_threshold=0.8,
                metadata={'goal': 'categorical_analysis'}
            ))
        
        return tasks
    
    def _apply_adaptation_rules(self, tasks: List[AnalysisTask], 
                              data_profile: Dict[str, Any]) -> List[AnalysisTask]:
        """Apply adaptation rules to modify the task list based on data characteristics"""
        adapted_tasks = tasks.copy()
        
        for rule_name, rule in self.adaptation_rules.items():
            if rule['condition'](data_profile):
                logger.info(f"Applying adaptation rule: {rule_name} - {rule['reason']}")
                adapted_tasks = self._apply_adaptation_action(adapted_tasks, rule['action'])
        
        return adapted_tasks
    
    def _apply_adaptation_action(self, tasks: List[AnalysisTask], 
                               action: str) -> List[AnalysisTask]:
        """Apply a specific adaptation action"""
        if action == 'add_clustering_analysis':
            # Add clustering if not already present
            if not any(task.strategy == AnalysisStrategy.CLUSTERING for task in tasks):
                tasks.append(AnalysisTask(
                    strategy=AnalysisStrategy.CLUSTERING,
                    priority=7,
                    dependencies=[AnalysisStrategy.BASIC_STATS],
                    estimated_time=120,
                    confidence_threshold=0.6,
                    metadata={'adaptive': True, 'reason': 'low_correlations'}
                ))
        
        elif action == 'prioritize_time_series_analysis':
            # Increase priority of time series analysis
            for task in tasks:
                if task.strategy == AnalysisStrategy.TIME_SERIES:
                    task.priority = 9
        
        elif action == 'add_dimensionality_reduction':
            # Add dimensionality reduction for high-dimensional data
            tasks.append(AnalysisTask(
                strategy=AnalysisStrategy.DIMENSIONALITY_REDUCTION,
                priority=8,
                dependencies=[AnalysisStrategy.CORRELATION_MATRIX],
                estimated_time=100,
                confidence_threshold=0.7,
                metadata={'adaptive': True, 'reason': 'high_dimensionality'}
            ))
        
        return tasks
    
    def _prioritize_and_order_tasks(self, tasks: List[AnalysisTask], 
                                   constraints: Dict[str, Any] = None) -> List[AnalysisTask]:
        """Prioritize and order tasks based on dependencies and constraints"""
        constraints = constraints or {}
        max_time = constraints.get('max_execution_time', 600)  # 10 minutes default
        
        # Sort by priority first, then by estimated time
        sorted_tasks = sorted(tasks, key=lambda t: (-t.priority, t.estimated_time))
        
        # Ensure dependencies are respected
        ordered_tasks = []
        completed_strategies = set()
        
        while sorted_tasks:
            # Find tasks that can be executed now
            executable_tasks = [task for task in sorted_tasks if task.can_execute(completed_strategies)]
            
            if not executable_tasks:
                # Break dependency deadlock by removing lowest priority task
                sorted_tasks.pop()
                continue
            
            # Select highest priority executable task
            selected_task = executable_tasks[0]
            ordered_tasks.append(selected_task)
            completed_strategies.add(selected_task.strategy)
            sorted_tasks.remove(selected_task)
            
            # Check time constraint
            total_time = sum(task.estimated_time for task in ordered_tasks)
            if total_time > max_time:
                logger.warning(f"Time constraint reached. Planned {len(ordered_tasks)} tasks.")
                break
        
        return ordered_tasks
    
    def _create_execution_plan(self, tasks: List[AnalysisTask], 
                             data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed execution plan with decision points and branching logic"""
        plan = {
            'analysis_id': data_profile.get('dataset_id', 'unknown'),
            'goals': [goal.value for goal in self.goals],
            'tasks': [],
            'decision_points': [],
            'recovery_strategies': [],
            'estimated_total_time': sum(task.estimated_time for task in tasks),
            'quality_gates': [],
            'metadata': {
                'data_profile': data_profile,
                'creation_time': 'now',
                'adaptive_rules_applied': len(self.adaptation_rules)
            }
        }
        
        for i, task in enumerate(tasks):
            task_dict = {
                'id': f"task_{i}",
                'strategy': task.strategy.value,
                'priority': task.priority,
                'dependencies': [dep.value for dep in task.dependencies],
                'estimated_time': task.estimated_time,
                'confidence_threshold': task.confidence_threshold,
                'metadata': task.metadata,
                'quality_checks': self._get_quality_checks_for_strategy(task.strategy),
                'fallback_strategies': self._get_fallback_strategies(task.strategy)
            }
            plan['tasks'].append(task_dict)
            
            # Add decision points after key tasks
            if task.strategy in [AnalysisStrategy.BASIC_STATS, 
                               AnalysisStrategy.CORRELATION_MATRIX,
                               AnalysisStrategy.DISTRIBUTION_ANALYSIS]:
                decision_point = {
                    'after_task': f"task_{i}",
                    'decision_type': 'adaptive_branching',
                    'conditions': self._get_decision_conditions(task.strategy),
                    'possible_actions': ['continue', 'add_task', 'skip_tasks', 'modify_parameters']
                }
                plan['decision_points'].append(decision_point)
        
        return plan
    
    def _add_quality_gates(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Add quality gates and monitoring checkpoints to the execution plan"""
        quality_gates = [
            {
                'checkpoint': 'data_quality_check',
                'after_task': 'task_0',  # After basic stats
                'criteria': {
                    'min_completeness': self.quality_thresholds['min_data_quality'],
                    'max_missing_pct': self.quality_thresholds['max_missing_data_pct']
                },
                'failure_action': 'trigger_data_cleaning'
            },
            {
                'checkpoint': 'analysis_confidence_check',
                'after_strategy': 'correlation_analysis',
                'criteria': {
                    'min_confidence': self.quality_thresholds['min_result_confidence']
                },
                'failure_action': 'try_alternative_methods'
            },
            {
                'checkpoint': 'result_variance_check',
                'after_strategy': 'distribution_analysis',
                'criteria': {
                    'min_variance': self.quality_thresholds['min_variance_threshold']
                },
                'failure_action': 'feature_engineering'
            }
        ]
        
        execution_plan['quality_gates'] = quality_gates
        return execution_plan
    
    def _get_quality_checks_for_strategy(self, strategy: AnalysisStrategy) -> List[str]:
        """Get quality checks specific to each analysis strategy"""
        checks = {
            AnalysisStrategy.BASIC_STATS: ['data_completeness', 'type_consistency'],
            AnalysisStrategy.CORRELATION_MATRIX: ['variance_check', 'multicollinearity'],
            AnalysisStrategy.DISTRIBUTION_ANALYSIS: ['normality_test', 'outlier_detection'],
            AnalysisStrategy.CLUSTERING: ['silhouette_score', 'elbow_method'],
            AnalysisStrategy.TIME_SERIES: ['stationarity_test', 'autocorrelation']
        }
        return checks.get(strategy, ['general_validity'])
    
    def _get_fallback_strategies(self, strategy: AnalysisStrategy) -> List[str]:
        """Get fallback strategies if the primary strategy fails"""
        fallbacks = {
            AnalysisStrategy.CORRELATION_MATRIX: ['spearman_correlation', 'mutual_information'],
            AnalysisStrategy.CLUSTERING: ['dbscan', 'gaussian_mixture'],
            AnalysisStrategy.ANOMALY_DETECTION: ['isolation_forest', 'local_outlier_factor'],
            AnalysisStrategy.TIME_SERIES: ['simple_decomposition', 'moving_averages']
        }
        return fallbacks.get(strategy, ['basic_fallback'])
    
    def _get_decision_conditions(self, strategy: AnalysisStrategy) -> List[Dict[str, Any]]:
        """Get decision conditions for adaptive branching"""
        conditions = {
            AnalysisStrategy.BASIC_STATS: [
                {
                    'condition': 'high_missing_data',
                    'threshold': 0.3,
                    'action': 'add_imputation_task'
                },
                {
                    'condition': 'many_categorical_features',
                    'threshold': 0.5,
                    'action': 'prioritize_categorical_analysis'
                }
            ],
            AnalysisStrategy.CORRELATION_MATRIX: [
                {
                    'condition': 'low_correlations',
                    'threshold': 0.3,
                    'action': 'add_clustering_analysis'
                },
                {
                    'condition': 'perfect_correlations',
                    'threshold': 0.95,
                    'action': 'remove_redundant_features'
                }
            ]
        }
        return conditions.get(strategy, [])
    
    # Strategy planning methods
    def _plan_basic_stats(self, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Plan basic statistics analysis"""
        return {
            'methods': ['describe', 'info', 'dtypes'],
            'visualizations': ['summary_table'],
            'focus_areas': ['missing_data', 'data_types', 'basic_distributions']
        }
    
    def _plan_correlation_analysis(self, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Plan correlation analysis"""
        numeric_cols = len(data_profile.get('column_types', {}).get('numerical', []))
        
        method = 'pearson'
        if numeric_cols > 20:
            method = 'spearman'  # More efficient for large datasets
        
        return {
            'method': method,
            'threshold': 0.5,
            'visualizations': ['heatmap', 'network_graph'],
            'focus_areas': ['strong_correlations', 'correlation_clusters']
        }
    
    def _plan_distribution_analysis(self, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Plan distribution analysis"""
        return {
            'methods': ['histogram', 'kde', 'qq_plot'],
            'tests': ['shapiro_wilk', 'kolmogorov_smirnov'],
            'visualizations': ['distribution_plots', 'box_plots'],
            'focus_areas': ['normality', 'skewness', 'outliers']
        }
    
    def _plan_clustering_analysis(self, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Plan clustering analysis"""
        rows = data_profile.get('basic_stats', {}).get('shape', {}).get('rows', 0)
        
        # Choose algorithm based on data size
        if rows < 1000:
            algorithm = 'hierarchical'
        elif rows < 10000:
            algorithm = 'kmeans'
        else:
            algorithm = 'mini_batch_kmeans'
        
        return {
            'algorithm': algorithm,
            'k_range': [2, 8],
            'preprocessing': ['standardization', 'pca'],
            'evaluation': ['silhouette', 'inertia'],
            'visualizations': ['cluster_scatter', 'dendrogram']
        }
    
    def _plan_anomaly_detection(self, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Plan anomaly detection"""
        return {
            'methods': ['isolation_forest', 'statistical_outliers'],
            'contamination': 0.1,
            'visualizations': ['outlier_scatter', 'anomaly_scores'],
            'validation': ['manual_inspection', 'domain_knowledge']
        }
    
    def _plan_time_series_analysis(self, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Plan time series analysis"""
        return {
            'components': ['trend', 'seasonality', 'residuals'],
            'tests': ['stationarity', 'autocorrelation'],
            'methods': ['decomposition', 'arima'],
            'visualizations': ['time_plot', 'decomposition_plot', 'acf_plot']
        }
    
    def _plan_categorical_analysis(self, data_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Plan categorical analysis"""
        return {
            'methods': ['frequency_tables', 'chi_square_tests'],
            'visualizations': ['bar_charts', 'contingency_tables'],
            'focus_areas': ['category_distributions', 'associations', 'rare_categories']
        }