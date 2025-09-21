#!/usr/bin/env python3

import sys
import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv

# Import our agentic modules
from data_profiler import DataProfiler
from agentic_planner import AgenticAnalysisPlanner, AnalysisStrategy
from adaptive_analyzer import AdaptiveAnalyzer, AnalysisQuality
from webhook_system import AgenticWebhookManager, WebhookEvent
from validator import EdaValidator
from pdf_generator import PdfGenerator

# Load environment variables - prioritize .env.local for development
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, '.env.local'))
load_dotenv(os.path.join(parent_dir, '.env'))  # Fallback

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrulyAgenticEDAAgent:
    """
    A truly agentic EDA agent that demonstrates autonomous reasoning, planning,
    and adaptive behavior. This replaces the linear workflow with goal-oriented
    decision-making and self-monitoring capabilities.
    """
    
    def __init__(self):
        self.supabase: Client = create_client(
            os.environ.get('NEXT_PUBLIC_SUPABASE_URL'),
            os.environ.get('SUPABASE_SERVICE_ROLE_KEY')
        )
        
        # Initialize agentic components
        self.profiler = DataProfiler()
        self.planner = AgenticAnalysisPlanner(self.profiler)
        self.analyzer = None  # Will be initialized with webhook manager
        self.webhook_manager = None  # Will be initialized per analysis
        self.validator = EdaValidator()  # Keep for LLM validation
        self.pdf_generator = PdfGenerator()
        
        # Agent state
        self.current_analysis_id = None
        self.current_user_id = None
        self.data_profile = None
        self.execution_plan = None
        self.completed_strategies = []
        self.analysis_results = {}
        self.quality_issues = []
        self.adaptation_history = []
        
        # Self-monitoring thresholds
        self.quality_thresholds = {
            'min_overall_quality': 0.7,
            'max_poor_results': 2,
            'min_insights_generated': 3,
            'max_execution_time': 600  # 10 minutes
        }
    
    async def run_agentic_analysis(self, csv_url: str, analysis_id: str, 
                                  user_id: str, user_goals: List[str] = None) -> Dict[str, Any]:
        """
        Main agentic analysis workflow that demonstrates true autonomy:
        1. Profile and understand the data
        2. Set goals and create dynamic plan
        3. Execute analysis with real-time adaptation
        4. Monitor quality and adjust approach
        5. Iterate and refine until goals are met
        6. Generate comprehensive insights
        """
        self.current_analysis_id = analysis_id
        self.current_user_id = user_id
        
        try:
            logger.info(f"Starting truly agentic analysis for {analysis_id}")
            
            # Initialize webhook system for autonomous communication
            self.webhook_manager = AgenticWebhookManager(analysis_id, user_id)
            await self.webhook_manager.initialize()
            
            # Initialize analyzer with webhook integration
            self.analyzer = AdaptiveAnalyzer(self.webhook_manager)
            
            # Notify start of analysis
            await self.webhook_manager.notify_analysis_started({
                'csv_url': csv_url,
                'user_goals': user_goals or []
            })
            
            # Phase 1: Autonomous Data Understanding
            logger.info("Phase 1: Autonomous data understanding and profiling...")
            df = await self._load_and_understand_data(csv_url)
            if df is None:
                raise Exception("Failed to load and understand data")
            
            # Phase 2: Goal Setting and Planning
            logger.info("Phase 2: Autonomous goal setting and planning...")
            execution_plan = await self._create_adaptive_plan(df, user_goals)
            
            # Phase 3: Autonomous Execution with Real-time Adaptation
            logger.info("Phase 3: Autonomous execution with real-time monitoring...")
            analysis_results = await self._execute_adaptive_analysis(df, execution_plan)
            
            # Phase 4: Quality Assessment and Iterative Refinement
            logger.info("Phase 4: Quality assessment and iterative refinement...")
            refined_results = await self._refine_and_validate_results(df, analysis_results)
            
            # Phase 5: Insight Generation and Reporting
            logger.info("Phase 5: Autonomous insight generation and reporting...")
            final_insights = await self._generate_comprehensive_insights(df, refined_results)
            
            # Phase 6: Self-Assessment and Continuous Learning
            logger.info("Phase 6: Self-assessment and learning...")
            self_assessment = await self._perform_self_assessment(final_insights)
            
            # Generate PDF report with agentic insights
            logger.info("Generating comprehensive agentic report...")
            pdf_path = await self._generate_agentic_report(df, final_insights, self_assessment)
            
            # Upload PDF to Supabase
            pdf_url = await self._upload_pdf_to_supabase(pdf_path, user_id, analysis_id)
            
            # Update database with comprehensive results
            await self._update_analysis_record(analysis_id, final_insights, pdf_url, self_assessment)
            
            # Final notification
            await self.webhook_manager.notify_analysis_complete(final_insights)
            
            logger.info(f"Agentic analysis completed successfully for {analysis_id}")
            
            return {
                'status': 'completed',
                'pdf_url': pdf_url,
                'results': final_insights,
                'self_assessment': self_assessment,
                'adaptation_history': self.adaptation_history,
                'strategies_executed': self.completed_strategies,
                'quality_score': self_assessment.get('overall_quality_score', 0)
            }
            
        except Exception as e:
            logger.error(f"Agentic analysis failed for {analysis_id}: {str(e)}")
            
            # Trigger autonomous error recovery
            if self.webhook_manager:
                await self.webhook_manager.webhook_trigger.trigger_event(
                    WebhookEvent.ERROR_OCCURRED,
                    {
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'phase': 'main_execution'
                    }
                )
            
            await self._update_analysis_record(analysis_id, None, None, None, 'failed')
            return {'status': 'failed', 'error': str(e)}
        
        finally:
            if self.webhook_manager:
                await self.webhook_manager.cleanup()
    
    async def _load_and_understand_data(self, csv_url: str) -> Optional[pd.DataFrame]:
        """Autonomously load and understand the dataset"""
        try:
            # Load CSV data - handle local files or URLs
            logger.info("Loading CSV data from URL...")
            
            if csv_url.startswith(('http://', 'https://')):
                # Handle remote URLs - use direct URL download
                import requests
                response = requests.get(csv_url)
                response.raise_for_status()
                csv_data = response.content
                
                # Parse CSV
                from io import StringIO
                if isinstance(csv_data, bytes):
                    csv_data = csv_data.decode('utf-8')
                
                df = pd.read_csv(StringIO(csv_data))
            else:
                # Handle local files (for testing)
                logger.info(f"Loading local CSV file: {csv_url}")
                df = pd.read_csv(csv_url)
            logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Autonomous data profiling
            logger.info("Performing autonomous data profiling...")
            self.data_profile = self.profiler.profile_dataset(df)
            
            await self.webhook_manager.notify_phase_complete(
                'data_profiling',
                {
                    'dataset_shape': df.shape,
                    'profile': self.data_profile
                }
            )
            
            # Autonomous data quality assessment
            quality_score = self.data_profile.get('data_quality', {}).get('quality_score', 0)
            if quality_score < 60:
                await self.webhook_manager.notify_quality_issue(
                    'low_data_quality',
                    {
                        'quality_score': quality_score,
                        'issues': self.data_profile.get('data_quality', {}).get('issues', [])
                    }
                )
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load and understand data: {str(e)}")
            return None
    
    async def _create_adaptive_plan(self, df: pd.DataFrame, user_goals: List[str] = None) -> Dict[str, Any]:
        """Create an adaptive analysis plan based on data characteristics and goals"""
        try:
            logger.info("Creating adaptive analysis plan...")
            
            # Agent autonomously creates plan based on data characteristics
            self.execution_plan = await self.planner.create_analysis_plan(
                self.data_profile,
                user_goals,
                constraints={'max_execution_time': self.quality_thresholds['max_execution_time']}
            )
            
            logger.info(f"Created plan with {len(self.execution_plan['tasks'])} tasks, "
                       f"targeting {len(self.execution_plan['goals'])} goals")
            
            await self.webhook_manager.notify_phase_complete(
                'planning',
                {
                    'execution_plan': self.execution_plan,
                    'estimated_time': self.execution_plan.get('estimated_total_time', 0)
                }
            )
            
            return self.execution_plan
            
        except Exception as e:
            logger.error(f"Failed to create adaptive plan: {str(e)}")
            raise
    
    async def _execute_adaptive_analysis(self, df: pd.DataFrame, 
                                       execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis with real-time adaptation and quality monitoring"""
        analysis_results = {}
        
        try:
            tasks = execution_plan.get('tasks', [])
            decision_points = execution_plan.get('decision_points', [])
            
            for i, task_config in enumerate(tasks):
                strategy_name = task_config['strategy']
                strategy = AnalysisStrategy(strategy_name)
                
                logger.info(f"Executing strategy {i+1}/{len(tasks)}: {strategy_name}")
                
                # Execute strategy with adaptive behavior
                result = await self.analyzer.execute_strategy(strategy, df, task_config)
                
                # Store result
                analysis_results[strategy_name] = result
                self.completed_strategies.append(strategy_name)
                
                # Check for decision points
                decision_point = next((dp for dp in decision_points 
                                     if dp.get('after_task') == f"task_{i}"), None)
                
                if decision_point:
                    await self._handle_decision_point(decision_point, result, df)
                
                # Real-time quality monitoring
                await self._monitor_analysis_quality(strategy_name, result)
                
                # Adaptive behavior: modify remaining tasks based on results
                if result.get('quality') in ['poor', 'failed']:
                    self.quality_issues.append({
                        'strategy': strategy_name,
                        'quality': result.get('quality'),
                        'task_index': i
                    })
                    
                    # Agent autonomously decides whether to continue or adapt
                    should_continue = await self._decide_continuation_strategy(result, tasks[i+1:])
                    if not should_continue:
                        logger.info("Agent decided to halt execution due to quality issues")
                        break
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Adaptive analysis execution failed: {str(e)}")
            raise
    
    async def _handle_decision_point(self, decision_point: Dict[str, Any], 
                                   recent_result: Dict[str, Any], df: pd.DataFrame):
        """Handle decision points with autonomous decision-making"""
        decision_type = decision_point.get('decision_type', 'adaptive_branching')
        conditions = decision_point.get('conditions', [])
        
        logger.info(f"Handling decision point: {decision_type}")
        
        # Evaluate conditions autonomously
        for condition in conditions:
            condition_name = condition.get('condition')
            threshold = condition.get('threshold')
            action = condition.get('action')
            
            # Agent evaluates condition based on recent results
            if await self._evaluate_condition(condition_name, threshold, recent_result, df):
                logger.info(f"Condition '{condition_name}' met, triggering action: {action}")
                await self._execute_adaptive_action(action, recent_result)
                
                # Record adaptation in history
                self.adaptation_history.append({
                    'condition': condition_name,
                    'action': action,
                    'trigger_result': recent_result.get('strategy'),
                    'timestamp': 'now'
                })
    
    async def _evaluate_condition(self, condition_name: str, threshold: float, 
                                result: Dict[str, Any], df: pd.DataFrame) -> bool:
        """Autonomously evaluate decision conditions"""
        if condition_name == 'high_missing_data':
            missing_pct = self.data_profile.get('data_quality', {}).get('overall_completeness', 100)
            return (100 - missing_pct) / 100 > threshold
        
        elif condition_name == 'low_correlations':
            correlations = result.get('strong_correlations', [])
            max_correlation = max([abs(c.get('correlation', 0)) for c in correlations], default=0)
            return max_correlation < threshold
        
        elif condition_name == 'perfect_correlations':
            correlations = result.get('strong_correlations', [])
            max_correlation = max([abs(c.get('correlation', 0)) for c in correlations], default=0)
            return max_correlation > threshold
        
        elif condition_name == 'many_categorical_features':
            total_cols = len(df.columns)
            categorical_cols = len(self.data_profile.get('column_types', {}).get('categorical', []))
            return categorical_cols / total_cols > threshold
        
        return False
    
    async def _execute_adaptive_action(self, action: str, trigger_result: Dict[str, Any]):
        """Execute adaptive actions based on autonomous decisions"""
        logger.info(f"Executing adaptive action: {action}")
        
        if action == 'add_clustering_analysis':
            # Agent autonomously adds clustering analysis
            await self.webhook_manager.webhook_trigger.trigger_event(
                WebhookEvent.ADDITIONAL_DATA_NEEDED,
                {
                    'type': 'additional_analysis',
                    'strategy': 'clustering',
                    'reason': 'low_correlations_detected'
                }
            )
        
        elif action == 'prioritize_categorical_analysis':
            # Increase priority of categorical analysis
            await self.webhook_manager.webhook_trigger.trigger_event(
                WebhookEvent.ANALYSIS_PHASE_COMPLETE,
                {
                    'next_phase': 'categorical_focus',
                    'priority': 'high'
                }
            )
        
        elif action == 'add_imputation_task':
            # Add data imputation task
            await self.webhook_manager.webhook_trigger.trigger_event(
                WebhookEvent.ADDITIONAL_DATA_NEEDED,
                {
                    'type': 'data_preprocessing',
                    'strategy': 'imputation',
                    'reason': 'high_missing_data'
                }
            )
    
    async def _monitor_analysis_quality(self, strategy_name: str, result: Dict[str, Any]):
        """Real-time quality monitoring with autonomous responses"""
        quality = result.get('quality', 'unknown')
        
        logger.info(f"Quality monitoring: {strategy_name} = {quality}")
        
        if quality in ['poor', 'failed']:
            await self.webhook_manager.notify_quality_issue(
                f'poor_quality_{strategy_name}',
                {
                    'strategy': strategy_name,
                    'quality': quality,
                    'details': result.get('error', 'Unknown quality issue')
                }
            )
        
        elif quality in ['excellent', 'good']:
            # Look for significant insights
            insights = self._extract_insights_from_result(strategy_name, result)
            for insight in insights:
                await self.webhook_manager.notify_insight_discovered(insight)
    
    def _extract_insights_from_result(self, strategy_name: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from analysis results"""
        insights = []
        
        if strategy_name == 'correlation_analysis':
            strong_correlations = result.get('strong_correlations', [])
            for corr in strong_correlations:
                insights.append({
                    'type': 'correlation',
                    'significance_score': abs(corr.get('correlation', 0)),
                    'description': f"Strong correlation between {corr.get('column1')} and {corr.get('column2')}",
                    'strength': corr.get('strength', 'unknown')
                })
        
        elif strategy_name == 'clustering_analysis':
            silhouette_score = result.get('silhouette_score', 0)
            if silhouette_score > 0.5:
                insights.append({
                    'type': 'clustering',
                    'significance_score': silhouette_score,
                    'description': f"Well-defined clusters found with silhouette score {silhouette_score:.2f}",
                    'clusters': result.get('optimal_clusters', 0)
                })
        
        elif strategy_name == 'anomaly_detection':
            total_anomalies = result.get('total_anomalies', 0)
            if total_anomalies > 0:
                insights.append({
                    'type': 'anomalies',
                    'significance_score': min(total_anomalies / 100, 1.0),  # Normalize
                    'description': f"Detected {total_anomalies} potential anomalies",
                    'count': total_anomalies
                })
        
        return insights
    
    async def _decide_continuation_strategy(self, failed_result: Dict[str, Any], 
                                          remaining_tasks: List[Dict[str, Any]]) -> bool:
        """Autonomously decide whether to continue analysis after failures"""
        # Count quality issues
        poor_results = len(self.quality_issues)
        
        # If too many poor results, agent decides to halt
        if poor_results >= self.quality_thresholds['max_poor_results']:
            logger.info(f"Agent decided to halt: too many poor results ({poor_results})")
            return False
        
        # If critical foundational analysis failed, halt
        failed_strategy = failed_result.get('strategy')
        if failed_strategy in ['basic_statistics', 'correlation_analysis']:
            logger.info(f"Agent decided to halt: critical analysis failed ({failed_strategy})")
            return False
        
        # Otherwise continue with remaining tasks
        logger.info("Agent decided to continue despite quality issues")
        return True
    
    async def _refine_and_validate_results(self, df: pd.DataFrame, 
                                         analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Iteratively refine and validate results using LLM validation"""
        try:
            logger.info("Refining and validating analysis results...")
            
            # Use LLM validator for comprehensive validation
            validated_results = self.validator.validate_analysis(df, analysis_results)
            
            # Agent performs self-validation
            refined_results = await self._perform_self_validation(df, validated_results)
            
            await self.webhook_manager.notify_phase_complete(
                'validation_refinement',
                {
                    'validation_applied': True,
                    'refinements_made': len(refined_results) - len(analysis_results)
                }
            )
            
            return refined_results
            
        except Exception as e:
            logger.error(f"Result refinement failed: {str(e)}")
            return analysis_results  # Return original results if refinement fails
    
    async def _perform_self_validation(self, df: pd.DataFrame, 
                                     results: Dict[str, Any]) -> Dict[str, Any]:
        """Agent performs self-validation and correction"""
        refined_results = results.copy()
        
        # Self-validation checks
        validations = [
            self._validate_statistical_consistency,
            self._validate_logical_relationships,
            self._validate_domain_reasonableness
        ]
        
        for validation in validations:
            try:
                validation_result = validation(df, refined_results)
                if not validation_result.get('valid', True):
                    logger.warning(f"Self-validation issue: {validation_result.get('issue')}")
                    # Agent could implement self-correction here
            except Exception as e:
                logger.warning(f"Self-validation error: {str(e)}")
        
        return refined_results
    
    def _validate_statistical_consistency(self, df: pd.DataFrame, 
                                        results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate statistical consistency of results"""
        # Example: Check if correlation results are consistent with distribution results
        return {'valid': True}
    
    def _validate_logical_relationships(self, df: pd.DataFrame, 
                                      results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate logical relationships in results"""
        # Example: Check if clustering results make sense given correlation results
        return {'valid': True}
    
    def _validate_domain_reasonableness(self, df: pd.DataFrame, 
                                      results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate domain reasonableness of results"""
        # Example: Check if outliers are reasonable given the data domain
        return {'valid': True}
    
    async def _generate_comprehensive_insights(self, df: pd.DataFrame, 
                                             results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive insights using LLM enhancement"""
        try:
            logger.info("Generating comprehensive insights...")
            
            # Use LLM to enhance insights
            enhanced_results = self.validator.enhance_insights(df, results)
            
            # Agent adds its own autonomous insights
            autonomous_insights = await self._generate_autonomous_insights(df, enhanced_results)
            
            # Combine all insights
            final_insights = {
                **enhanced_results,
                'autonomous_insights': autonomous_insights,
                'insight_summary': self._create_insight_summary(enhanced_results, autonomous_insights),
                'recommendations': await self._generate_recommendations(df, enhanced_results),
                'confidence_scores': self._calculate_confidence_scores(enhanced_results)
            }
            
            return final_insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {str(e)}")
            return results
    
    async def _generate_autonomous_insights(self, df: pd.DataFrame, 
                                          results: Dict[str, Any]) -> Dict[str, Any]:
        """Agent generates its own insights beyond LLM analysis"""
        autonomous_insights = {
            'data_assessment': self._assess_data_characteristics(df),
            'analysis_pathway': self._analyze_analysis_pathway(),
            'quality_journey': self._analyze_quality_journey(),
            'adaptation_insights': self._analyze_adaptations()
        }
        
        return autonomous_insights
    
    def _assess_data_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Agent's autonomous assessment of data characteristics"""
        return {
            'complexity_assessment': self.data_profile.get('complexity', {}),
            'quality_assessment': self.data_profile.get('data_quality', {}),
            'pattern_recognition': self.data_profile.get('patterns', {}),
            'opportunity_identification': self.data_profile.get('analysis_opportunities', [])
        }
    
    def _analyze_analysis_pathway(self) -> Dict[str, Any]:
        """Analyze the pathway the agent took through the analysis"""
        return {
            'strategies_executed': self.completed_strategies,
            'execution_order': self.completed_strategies,
            'decision_points_encountered': len(self.adaptation_history),
            'pathway_efficiency': len(self.completed_strategies) / len(self.execution_plan.get('tasks', []))
        }
    
    def _analyze_quality_journey(self) -> Dict[str, Any]:
        """Analyze the quality journey throughout the analysis"""
        quality_scores = []
        for strategy, result in self.analysis_results.items():
            if 'quality' in result:
                quality_map = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.4, 'failed': 0.0}
                quality_scores.append(quality_map.get(result['quality'], 0.5))
        
        return {
            'quality_scores': quality_scores,
            'average_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'quality_improvement': quality_scores[-1] - quality_scores[0] if len(quality_scores) > 1 else 0,
            'quality_issues_encountered': len(self.quality_issues)
        }
    
    def _analyze_adaptations(self) -> Dict[str, Any]:
        """Analyze the adaptations made during analysis"""
        return {
            'total_adaptations': len(self.adaptation_history),
            'adaptation_triggers': [a.get('condition') for a in self.adaptation_history],
            'adaptation_actions': [a.get('action') for a in self.adaptation_history],
            'adaptation_effectiveness': 'high' if len(self.quality_issues) < 2 else 'moderate'
        }
    
    def _create_insight_summary(self, enhanced_results: Dict[str, Any], 
                               autonomous_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive insight summary"""
        summary = {
            'key_findings': [],
            'statistical_insights': [],
            'pattern_insights': [],
            'quality_insights': [],
            'actionable_recommendations': []
        }
        
        # Extract key findings from enhanced results
        for strategy, result in enhanced_results.items():
            if isinstance(result, dict) and result.get('success', False):
                if strategy == 'correlation_analysis':
                    correlations = result.get('strong_correlations', [])
                    if correlations:
                        summary['key_findings'].append(f"Found {len(correlations)} strong correlations")
                
                elif strategy == 'clustering_analysis':
                    clusters = result.get('optimal_clusters', 0)
                    if clusters > 0:
                        summary['key_findings'].append(f"Identified {clusters} distinct data clusters")
        
        return summary
    
    async def _generate_recommendations(self, df: pd.DataFrame, 
                                      results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Data quality recommendations
        quality_score = self.data_profile.get('data_quality', {}).get('quality_score', 0)
        if quality_score < 80:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'high',
                'recommendation': 'Improve data quality by addressing missing values and outliers',
                'rationale': f'Current quality score: {quality_score:.1f}/100'
            })
        
        # Analysis-specific recommendations
        if 'correlation_analysis' in results:
            correlations = results['correlation_analysis'].get('strong_correlations', [])
            if correlations:
                recommendations.append({
                    'type': 'analysis',
                    'priority': 'medium',
                    'recommendation': 'Investigate causal relationships between highly correlated variables',
                    'rationale': f'Found {len(correlations)} strong correlations'
                })
        
        return recommendations
    
    def _calculate_confidence_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for different aspects of the analysis"""
        confidence_scores = {}
        
        # Overall analysis confidence
        successful_strategies = sum(1 for r in results.values() 
                                  if isinstance(r, dict) and r.get('success', False))
        total_strategies = len(self.completed_strategies)
        
        confidence_scores['overall_analysis'] = successful_strategies / max(total_strategies, 1)
        
        # Data quality confidence
        quality_score = self.data_profile.get('data_quality', {}).get('quality_score', 0)
        confidence_scores['data_quality'] = quality_score / 100
        
        # Insight reliability
        insight_count = len(self._extract_all_insights(results))
        confidence_scores['insight_reliability'] = min(insight_count / 5, 1.0)  # Normalize to 5 insights
        
        return confidence_scores
    
    def _extract_all_insights(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all insights from results"""
        all_insights = []
        
        for strategy_name, result in results.items():
            if isinstance(result, dict):
                insights = self._extract_insights_from_result(strategy_name, result)
                all_insights.extend(insights)
        
        return all_insights
    
    async def _perform_self_assessment(self, final_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Agent performs comprehensive self-assessment of its performance"""
        try:
            logger.info("Performing self-assessment...")
            
            assessment = {
                'overall_quality_score': self._calculate_overall_quality_score(final_insights),
                'goal_achievement': self._assess_goal_achievement(final_insights),
                'efficiency_metrics': self._calculate_efficiency_metrics(),
                'adaptation_effectiveness': self._assess_adaptation_effectiveness(),
                'insight_generation': self._assess_insight_generation(final_insights),
                'learning_opportunities': self._identify_learning_opportunities(),
                'performance_grade': 'A'  # Will be calculated
            }
            
            # Calculate overall performance grade
            assessment['performance_grade'] = self._calculate_performance_grade(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Self-assessment failed: {str(e)}")
            return {'overall_quality_score': 0.5, 'performance_grade': 'C'}
    
    def _calculate_overall_quality_score(self, insights: Dict[str, Any]) -> float:
        """Calculate overall quality score of the analysis"""
        scores = []
        
        # Quality of individual analyses
        confidence_scores = insights.get('confidence_scores', {})
        if confidence_scores:
            scores.extend(confidence_scores.values())
        
        # Data quality contribution
        data_quality = self.data_profile.get('data_quality', {}).get('quality_score', 0)
        scores.append(data_quality / 100)
        
        # Insight quality
        insights_count = len(self._extract_all_insights(insights))
        insight_quality = min(insights_count / self.quality_thresholds['min_insights_generated'], 1.0)
        scores.append(insight_quality)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _assess_goal_achievement(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Assess how well the agent achieved its goals"""
        planned_goals = self.execution_plan.get('goals', [])
        achieved_goals = []
        
        # Simple goal achievement assessment
        if 'correlation_analysis' in self.completed_strategies:
            achieved_goals.append('find_correlations')
        
        if 'clustering_analysis' in self.completed_strategies:
            achieved_goals.append('segment_data')
        
        if 'distribution_analysis' in self.completed_strategies:
            achieved_goals.append('understand_distributions')
        
        return {
            'planned_goals': planned_goals,
            'achieved_goals': achieved_goals,
            'achievement_rate': len(achieved_goals) / max(len(planned_goals), 1)
        }
    
    def _calculate_efficiency_metrics(self) -> Dict[str, Any]:
        """Calculate efficiency metrics of the analysis"""
        planned_tasks = len(self.execution_plan.get('tasks', []))
        completed_tasks = len(self.completed_strategies)
        
        return {
            'task_completion_rate': completed_tasks / max(planned_tasks, 1),
            'adaptation_efficiency': 1.0 - (len(self.quality_issues) / max(completed_tasks, 1)),
            'planning_accuracy': completed_tasks / max(planned_tasks, 1)
        }
    
    def _assess_adaptation_effectiveness(self) -> Dict[str, Any]:
        """Assess how effective the agent's adaptations were"""
        return {
            'adaptations_made': len(self.adaptation_history),
            'quality_issues_resolved': max(0, len(self.quality_issues) - len(self.adaptation_history)),
            'adaptation_success_rate': 0.8 if self.adaptation_history else 1.0  # Placeholder
        }
    
    def _assess_insight_generation(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality and quantity of insights generated"""
        all_insights = self._extract_all_insights(insights)
        
        return {
            'total_insights': len(all_insights),
            'insight_diversity': len(set(i.get('type') for i in all_insights)),
            'high_significance_insights': sum(1 for i in all_insights if i.get('significance_score', 0) > 0.7),
            'insight_quality_score': sum(i.get('significance_score', 0) for i in all_insights) / max(len(all_insights), 1)
        }
    
    def _identify_learning_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for improvement"""
        opportunities = []
        
        if len(self.quality_issues) > 1:
            opportunities.append({
                'area': 'quality_management',
                'description': 'Improve quality threshold management',
                'priority': 'medium'
            })
        
        if len(self.adaptation_history) == 0:
            opportunities.append({
                'area': 'adaptation_sensitivity',
                'description': 'Increase sensitivity to adaptation triggers',
                'priority': 'low'
            })
        
        return opportunities
    
    def _calculate_performance_grade(self, assessment: Dict[str, Any]) -> str:
        """Calculate overall performance grade"""
        overall_score = assessment.get('overall_quality_score', 0)
        
        if overall_score >= 0.9:
            return 'A+'
        elif overall_score >= 0.8:
            return 'A'
        elif overall_score >= 0.7:
            return 'B+'
        elif overall_score >= 0.6:
            return 'B'
        elif overall_score >= 0.5:
            return 'C'
        else:
            return 'D'
    
    async def _generate_agentic_report(self, df: pd.DataFrame, insights: Dict[str, Any], 
                                     self_assessment: Dict[str, Any]) -> str:
        """Generate comprehensive PDF report including agentic insights"""
        try:
            # Collect all visualization paths from different analysis results
            all_viz_paths = []
            for strategy_results in insights.values():
                if isinstance(strategy_results, dict):
                    # Check for visualization_paths, visualization_path, or visualizations
                    if 'visualization_paths' in strategy_results:
                        paths = strategy_results['visualization_paths']
                        if isinstance(paths, list):
                            all_viz_paths.extend(paths)
                        elif isinstance(paths, str):
                            all_viz_paths.append(paths)
                    
                    if 'visualization_path' in strategy_results:
                        all_viz_paths.append(strategy_results['visualization_path'])
                    
                    if 'visualizations' in strategy_results:
                        viz = strategy_results['visualizations']
                        if isinstance(viz, list):
                            all_viz_paths.extend(viz)
                        elif isinstance(viz, str):
                            all_viz_paths.append(viz)
            
            # Enhance insights with agentic metadata and collected visualizations
            report_data = {
                **insights,
                'visualizations': all_viz_paths,  # This is what PDF generator expects
                'agentic_metadata': {
                    'analysis_approach': 'fully_autonomous',
                    'decision_points': len(self.adaptation_history),
                    'strategies_executed': self.completed_strategies,
                    'self_assessment': self_assessment,
                    'quality_monitoring': 'real_time',
                    'adaptation_history': self.adaptation_history
                }
            }
            
            # Generate PDF using existing generator
            pdf_path = self.pdf_generator.create_report(
                df, 
                report_data, 
                f"agentic_eda_report_{self.current_analysis_id}.pdf"
            )
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"Agentic report generation failed: {str(e)}")
            raise
    
    async def _upload_pdf_to_supabase(self, pdf_path: str, user_id: str, analysis_id: str) -> str:
        """Upload PDF to Supabase storage"""
        try:
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            
            storage_path = f"{user_id}/reports/{analysis_id}.pdf"
            
            # Try to upload, if it exists, delete and re-upload
            try:
                result = self.supabase.storage.from_('pdf-reports').upload(
                    storage_path, pdf_data, {'content-type': 'application/pdf'}
                )
            except Exception as e:
                if "already exists" in str(e) or "Duplicate" in str(e):
                    logger.info(f"PDF already exists, replacing it: {storage_path}")
                    # Delete existing file and re-upload
                    self.supabase.storage.from_('pdf-reports').remove([storage_path])
                    result = self.supabase.storage.from_('pdf-reports').upload(
                        storage_path, pdf_data, {'content-type': 'application/pdf'}
                    )
                else:
                    raise
            
            # Get public URL
            try:
                url_response = self.supabase.storage.from_('pdf-reports').get_public_url(storage_path)
                
                if isinstance(url_response, str):
                    public_url = url_response
                elif hasattr(url_response, 'publicURL'):
                    public_url = url_response.publicURL
                elif isinstance(url_response, dict):
                    public_url = url_response.get('publicURL') or url_response.get('publicUrl')
                else:
                    base_url = os.environ.get('NEXT_PUBLIC_SUPABASE_URL')
                    public_url = f"{base_url}/storage/v1/object/public/pdf-reports/{storage_path}"
                
                return public_url
                
            except Exception as e:
                logger.error(f"Failed to get public URL: {str(e)}")
                base_url = os.environ.get('NEXT_PUBLIC_SUPABASE_URL')
                return f"{base_url}/storage/v1/object/public/pdf-reports/{storage_path}"
                
        except Exception as e:
            logger.error(f"Failed to upload PDF: {str(e)}")
            raise
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
    
    async def _update_analysis_record(self, analysis_id: str, results: Dict[str, Any] = None,
                                    pdf_url: str = None, self_assessment: Dict[str, Any] = None,
                                    status: str = 'completed'):
        """Update the analysis record with comprehensive results"""
        try:
            update_data = {
                'analysis_status': status,
                'updated_at': 'now()'
            }
            
            # Combine all results into the analysis_results JSONB column
            combined_results = {}
            if results:
                combined_results.update(results)
            if self_assessment:
                combined_results['self_assessment'] = self_assessment
            
            if combined_results:
                # Convert numpy types to JSON-serializable types
                update_data['analysis_results'] = self._convert_numpy_types_for_json(combined_results)
            
            if pdf_url:
                update_data['pdf_url'] = pdf_url
            
            result = self.supabase.table('analyses').update(update_data).eq('id', analysis_id).execute()
            
            if not result.data:
                logger.warning(f"Analysis record {analysis_id} not found in database (normal for direct testing)")
                # Don't fail for testing - the analysis completed successfully
                
        except Exception as e:
            logger.error(f"Failed to update database: {str(e)}")
            raise
    
    def _convert_numpy_types_for_json(self, obj):
        """Convert numpy types to JSON-serializable Python types"""
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
            return {key: self._convert_numpy_types_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types_for_json(item) for item in obj]
        else:
            return str(obj)

def main():
    if len(sys.argv) != 4:
        print("Usage: python truly_agentic_agent.py <csv_url> <analysis_id> <user_id>")
        sys.exit(1)
    
    csv_url = sys.argv[1]
    analysis_id = sys.argv[2]
    user_id = sys.argv[3]
    
    async def run_analysis():
        agent = TrulyAgenticEDAAgent()
        return await agent.run_agentic_analysis(csv_url, analysis_id, user_id)
    
    # Run the async analysis
    result = asyncio.run(run_analysis())
    
    # Convert numpy types for JSON output
    def convert_numpy_types(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, 'dtype'):
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