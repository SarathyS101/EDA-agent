#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import os
from enum import Enum

logger = logging.getLogger(__name__)

class WebhookEvent(Enum):
    """Types of events that can trigger webhooks"""
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_PHASE_COMPLETE = "analysis.phase_complete"
    QUALITY_CHECK_FAILED = "quality.check_failed"
    ADDITIONAL_DATA_NEEDED = "data.additional_needed"
    INSIGHT_DISCOVERED = "insight.discovered"
    ANALYSIS_COMPLETE = "analysis.complete"
    ERROR_OCCURRED = "error.occurred"
    RECOVERY_INITIATED = "recovery.initiated"

class WebhookTrigger:
    """
    Autonomous webhook system that allows the agent to communicate with external systems
    and trigger its own actions based on analysis results.
    """
    
    def __init__(self, analysis_id: str, base_url: str = None):
        self.analysis_id = analysis_id
        self.base_url = base_url or os.environ.get('NEXTAUTH_URL', 'http://localhost:3000')
        self.session = None
        self.registered_hooks = {}
        self.event_history = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def register_hook(self, event: WebhookEvent, callback: Callable = None, 
                     external_url: str = None, retry_attempts: int = 3):
        """Register a webhook for specific events"""
        hook_config = {
            'callback': callback,
            'external_url': external_url,
            'retry_attempts': retry_attempts,
            'created_at': datetime.now().isoformat()
        }
        
        if event not in self.registered_hooks:
            self.registered_hooks[event] = []
        
        self.registered_hooks[event].append(hook_config)
        logger.info(f"Registered webhook for {event.value}")
    
    async def trigger_event(self, event: WebhookEvent, data: Dict[str, Any] = None, 
                          metadata: Dict[str, Any] = None):
        """Trigger an event and execute all registered webhooks"""
        event_data = {
            'event_type': event.value,
            'analysis_id': self.analysis_id,
            'timestamp': datetime.now().isoformat(),
            'data': data or {},
            'metadata': metadata or {}
        }
        
        # Store event in history
        self.event_history.append(event_data)
        
        logger.info(f"Triggering event: {event.value} for analysis {self.analysis_id}")
        
        # Execute registered hooks
        if event in self.registered_hooks:
            tasks = []
            for hook_config in self.registered_hooks[event]:
                tasks.append(self._execute_hook(hook_config, event_data))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_hook(self, hook_config: Dict[str, Any], event_data: Dict[str, Any]):
        """Execute a single webhook"""
        try:
            # Execute callback if provided
            if hook_config.get('callback'):
                await self._execute_callback(hook_config['callback'], event_data)
            
            # Make external HTTP request if URL provided
            if hook_config.get('external_url'):
                await self._make_http_request(
                    hook_config['external_url'], 
                    event_data,
                    hook_config.get('retry_attempts', 3)
                )
                
        except Exception as e:
            logger.error(f"Hook execution failed: {str(e)}")
            
            # Trigger error recovery
            await self.trigger_event(
                WebhookEvent.ERROR_OCCURRED,
                {'error': str(e), 'hook_config': hook_config},
                {'original_event': event_data}
            )
    
    async def _execute_callback(self, callback: Callable, event_data: Dict[str, Any]):
        """Execute a callback function"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event_data)
            else:
                callback(event_data)
        except Exception as e:
            logger.error(f"Callback execution failed: {str(e)}")
            raise
    
    async def _make_http_request(self, url: str, data: Dict[str, Any], 
                               retry_attempts: int = 3):
        """Make HTTP request with retry logic"""
        last_exception = None
        
        for attempt in range(retry_attempts):
            try:
                if not self.session:
                    self.session = aiohttp.ClientSession()
                
                async with self.session.post(
                    url,
                    json=data,
                    headers={'Content-Type': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook sent successfully to {url}")
                        return await response.json()
                    else:
                        logger.warning(f"Webhook returned status {response.status}: {await response.text()}")
                        
            except Exception as e:
                last_exception = e
                logger.warning(f"Webhook attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise last_exception or Exception("All webhook attempts failed")

class AgenticWebhookManager:
    """
    High-level webhook manager that provides agentic capabilities for
    self-monitoring, external communication, and autonomous decision-making.
    """
    
    def __init__(self, analysis_id: str, user_id: str):
        self.analysis_id = analysis_id
        self.user_id = user_id
        self.webhook_trigger = None
        self.decision_hooks = {}
        self.monitoring_active = False
        
    async def initialize(self):
        """Initialize the webhook system"""
        self.webhook_trigger = WebhookTrigger(self.analysis_id)
        await self.webhook_trigger.__aenter__()
        
        # Register core agentic hooks
        await self._setup_agentic_hooks()
        
        logger.info(f"Agentic webhook manager initialized for analysis {self.analysis_id}")
    
    async def cleanup(self):
        """Clean up resources"""
        if self.webhook_trigger:
            await self.webhook_trigger.__aexit__(None, None, None)
    
    async def _setup_agentic_hooks(self):
        """Setup hooks that enable autonomous agent behavior"""
        
        # Hook for autonomous quality monitoring
        self.webhook_trigger.register_hook(
            WebhookEvent.QUALITY_CHECK_FAILED,
            callback=self._handle_quality_failure
        )
        
        # Hook for requesting additional analysis when insights are unclear
        self.webhook_trigger.register_hook(
            WebhookEvent.INSIGHT_DISCOVERED,
            callback=self._evaluate_insight_significance
        )
        
        # Hook for autonomous recovery from errors
        self.webhook_trigger.register_hook(
            WebhookEvent.ERROR_OCCURRED,
            callback=self._initiate_recovery
        )
        
        # Hook for triggering next analysis phase
        self.webhook_trigger.register_hook(
            WebhookEvent.ANALYSIS_PHASE_COMPLETE,
            callback=self._decide_next_phase
        )
        
        # External notification hooks
        base_url = os.environ.get('NEXTAUTH_URL', 'http://localhost:3000')
        
        self.webhook_trigger.register_hook(
            WebhookEvent.ANALYSIS_STARTED,
            external_url=f"{base_url}/api/webhook/analysis-started"
        )
        
        self.webhook_trigger.register_hook(
            WebhookEvent.ANALYSIS_COMPLETE,
            external_url=f"{base_url}/api/webhook/analysis-complete"
        )
        
        # Integration hooks for external data sources
        self.webhook_trigger.register_hook(
            WebhookEvent.ADDITIONAL_DATA_NEEDED,
            callback=self._request_additional_data
        )
    
    async def _handle_quality_failure(self, event_data: Dict[str, Any]):
        """Autonomous handler for quality check failures"""
        quality_issue = event_data['data'].get('quality_issue')
        logger.info(f"Handling quality failure: {quality_issue}")
        
        # Decide on recovery strategy based on the issue
        if quality_issue == 'high_missing_data':
            await self._trigger_missing_data_strategy(event_data)
        elif quality_issue == 'low_correlation':
            await self._trigger_alternative_analysis(event_data)
        elif quality_issue == 'insufficient_variance':
            await self._trigger_feature_engineering(event_data)
        else:
            await self.webhook_trigger.trigger_event(
                WebhookEvent.RECOVERY_INITIATED,
                {'strategy': 'generic_fallback', 'original_issue': quality_issue}
            )
    
    async def _evaluate_insight_significance(self, event_data: Dict[str, Any]):
        """Evaluate if discovered insights warrant additional analysis"""
        insight = event_data['data'].get('insight', {})
        significance = insight.get('significance_score', 0)
        
        if significance > 0.8:  # High significance
            logger.info("High-significance insight discovered, triggering deep dive analysis")
            await self._trigger_deep_dive_analysis(insight)
        elif significance > 0.5:  # Medium significance
            logger.info("Medium-significance insight discovered, requesting validation")
            await self._request_insight_validation(insight)
        else:
            logger.info("Low-significance insight, continuing with standard analysis")
    
    async def _initiate_recovery(self, event_data: Dict[str, Any]):
        """Autonomous error recovery system"""
        error_data = event_data['data']
        error_type = error_data.get('error_type', 'unknown')
        
        logger.info(f"Initiating recovery for error type: {error_type}")
        
        recovery_strategies = {
            'data_loading_error': self._recover_data_loading,
            'analysis_timeout': self._recover_analysis_timeout,
            'memory_error': self._recover_memory_error,
            'api_rate_limit': self._recover_api_limit
        }
        
        recovery_func = recovery_strategies.get(error_type, self._generic_recovery)
        await recovery_func(event_data)
    
    async def _decide_next_phase(self, event_data: Dict[str, Any]):
        """Autonomously decide what analysis phase to run next"""
        completed_phase = event_data['data'].get('completed_phase')
        results = event_data['data'].get('results', {})
        
        logger.info(f"Completed phase: {completed_phase}, deciding next phase")
        
        # Decision logic based on results
        if completed_phase == 'basic_profiling':
            if results.get('complexity', {}).get('level') == 'complex':
                await self._trigger_advanced_analysis()
            else:
                await self._trigger_standard_analysis()
                
        elif completed_phase == 'correlation_analysis':
            if results.get('strong_correlations', []):
                await self._trigger_causal_analysis()
            else:
                await self._trigger_clustering_analysis()
        
        elif completed_phase == 'clustering':
            await self._trigger_final_reporting()
    
    async def _trigger_missing_data_strategy(self, event_data: Dict[str, Any]):
        """Handle missing data through various strategies"""
        strategies = ['imputation', 'removal', 'alternative_features']
        
        for strategy in strategies:
            await self.webhook_trigger.trigger_event(
                WebhookEvent.RECOVERY_INITIATED,
                {
                    'strategy': f'missing_data_{strategy}',
                    'automated': True
                }
            )
    
    async def _trigger_alternative_analysis(self, event_data: Dict[str, Any]):
        """Trigger alternative analysis methods"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.RECOVERY_INITIATED,
            {
                'strategy': 'alternative_analysis',
                'methods': ['non_parametric', 'categorical_focus', 'text_analysis']
            }
        )
    
    async def _trigger_feature_engineering(self, event_data: Dict[str, Any]):
        """Autonomously trigger feature engineering"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.ADDITIONAL_DATA_NEEDED,
            {
                'type': 'feature_engineering',
                'methods': ['polynomial_features', 'interaction_terms', 'transformations']
            }
        )
    
    async def _trigger_deep_dive_analysis(self, insight: Dict[str, Any]):
        """Trigger deep dive analysis for significant insights"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.ANALYSIS_PHASE_COMPLETE,
            {
                'next_phase': 'deep_dive',
                'focus_area': insight.get('category'),
                'automated_trigger': True
            }
        )
    
    async def _request_insight_validation(self, insight: Dict[str, Any]):
        """Request validation for medium-significance insights"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.ADDITIONAL_DATA_NEEDED,
            {
                'type': 'validation',
                'insight': insight,
                'validation_methods': ['bootstrap', 'cross_validation']
            }
        )
    
    async def _request_additional_data(self, event_data: Dict[str, Any]):
        """Request additional data sources or features"""
        data_type = event_data['data'].get('type')
        
        if data_type == 'external_enrichment':
            # Could integrate with external APIs here
            logger.info("Would integrate with external data sources")
        elif data_type == 'feature_engineering':
            # Trigger automated feature creation
            logger.info("Triggering automated feature engineering")
        
        # For now, log the request - in a full implementation,
        # this would integrate with external data sources
        logger.info(f"Additional data requested: {event_data['data']}")
    
    async def _recover_data_loading(self, event_data: Dict[str, Any]):
        """Recover from data loading errors"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.RECOVERY_INITIATED,
            {'strategy': 'retry_data_loading', 'fallback': 'sample_analysis'}
        )
    
    async def _recover_analysis_timeout(self, event_data: Dict[str, Any]):
        """Recover from analysis timeouts"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.RECOVERY_INITIATED,
            {'strategy': 'reduce_complexity', 'sample_data': True}
        )
    
    async def _recover_memory_error(self, event_data: Dict[str, Any]):
        """Recover from memory errors"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.RECOVERY_INITIATED,
            {'strategy': 'chunk_processing', 'reduce_features': True}
        )
    
    async def _recover_api_limit(self, event_data: Dict[str, Any]):
        """Recover from API rate limits"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.RECOVERY_INITIATED,
            {'strategy': 'exponential_backoff', 'use_cache': True}
        )
    
    async def _generic_recovery(self, event_data: Dict[str, Any]):
        """Generic recovery strategy"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.RECOVERY_INITIATED,
            {'strategy': 'fallback_analysis', 'simplified': True}
        )
    
    async def _trigger_advanced_analysis(self):
        """Trigger advanced analysis for complex datasets"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.ANALYSIS_PHASE_COMPLETE,
            {'next_phase': 'advanced', 'methods': ['dimensionality_reduction', 'clustering']}
        )
    
    async def _trigger_standard_analysis(self):
        """Trigger standard analysis for simpler datasets"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.ANALYSIS_PHASE_COMPLETE,
            {'next_phase': 'standard', 'methods': ['correlation', 'distribution']}
        )
    
    async def _trigger_causal_analysis(self):
        """Trigger causal analysis when strong correlations are found"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.ANALYSIS_PHASE_COMPLETE,
            {'next_phase': 'causal', 'methods': ['granger_causality', 'structural_equations']}
        )
    
    async def _trigger_clustering_analysis(self):
        """Trigger clustering when correlations are weak"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.ANALYSIS_PHASE_COMPLETE,
            {'next_phase': 'clustering', 'methods': ['kmeans', 'hierarchical']}
        )
    
    async def _trigger_final_reporting(self):
        """Trigger final report generation"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.ANALYSIS_COMPLETE,
            {'final_phase': True, 'report_ready': True}
        )
    
    # Convenience methods for common webhook triggers
    async def notify_analysis_started(self, metadata: Dict[str, Any] = None):
        """Notify that analysis has started"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.ANALYSIS_STARTED,
            {'user_id': self.user_id},
            metadata
        )
    
    async def notify_phase_complete(self, phase_name: str, results: Dict[str, Any]):
        """Notify that an analysis phase is complete"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.ANALYSIS_PHASE_COMPLETE,
            {
                'completed_phase': phase_name,
                'results': results,
                'user_id': self.user_id
            }
        )
    
    async def notify_quality_issue(self, issue_type: str, details: Dict[str, Any]):
        """Notify about quality issues"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.QUALITY_CHECK_FAILED,
            {
                'quality_issue': issue_type,
                'details': details,
                'user_id': self.user_id
            }
        )
    
    async def notify_insight_discovered(self, insight: Dict[str, Any]):
        """Notify about discovered insights"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.INSIGHT_DISCOVERED,
            {
                'insight': insight,
                'user_id': self.user_id
            }
        )
    
    async def notify_analysis_complete(self, final_results: Dict[str, Any]):
        """Notify that analysis is complete"""
        await self.webhook_trigger.trigger_event(
            WebhookEvent.ANALYSIS_COMPLETE,
            {
                'results': final_results,
                'user_id': self.user_id
            }
        )