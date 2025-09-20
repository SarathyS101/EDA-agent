import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class PdfGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_JUSTIFY
        )
    
    def create_report(self, df: pd.DataFrame, analysis_results: Dict[str, Any], filename: str) -> str:
        """Generate comprehensive PDF report"""
        try:
            doc = SimpleDocTemplate(filename, pagesize=A4)
            story = []
            
            # Title page
            story.extend(self._create_title_page(df))
            
            # Executive summary
            story.extend(self._create_executive_summary(df, analysis_results))
            story.append(PageBreak())
            
            # Data overview
            story.extend(self._create_data_overview(df, analysis_results))
            story.append(PageBreak())
            
            # Statistical analysis
            story.extend(self._create_statistical_analysis(analysis_results))
            story.append(PageBreak())
            
            # Data quality analysis
            story.extend(self._create_data_quality_section(analysis_results))
            story.append(PageBreak())
            
            # Insights and recommendations
            story.extend(self._create_insights_section(analysis_results))
            story.append(PageBreak())
            
            # Visualizations
            if 'visualizations' in analysis_results:
                story.extend(self._create_visualizations_section(analysis_results['visualizations']))
            
            # Build PDF
            doc.build(story)
            logger.info(f"PDF report generated successfully: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {str(e)}")
            raise
    
    def _create_title_page(self, df: pd.DataFrame) -> List:
        """Create title page"""
        story = []
        
        # Main title
        story.append(Paragraph("Exploratory Data Analysis Report", self.title_style))
        story.append(Spacer(1, 0.5 * inch))
        
        # Dataset info
        story.append(Paragraph("Dataset Overview", self.heading_style))
        
        dataset_info = [
            ["Metric", "Value"],
            ["Total Rows", f"{df.shape[0]:,}"],
            ["Total Columns", f"{df.shape[1]:,}"],
            ["Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"],
            ["Analysis Date", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        table = Table(dataset_info, colWidths=[2 * inch, 2 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(PageBreak())
        
        return story
    
    def _create_executive_summary(self, df: pd.DataFrame, results: Dict[str, Any]) -> List:
        """Create executive summary"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.title_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Key findings
        if 'insights' in results:
            story.append(Paragraph("Key Findings:", self.heading_style))
            for insight in results['insights'][:5]:  # Top 5 insights
                story.append(Paragraph(f"• {insight}", self.normal_style))
        
        story.append(Spacer(1, 0.2 * inch))
        
        # Data quality summary
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        duplicate_pct = (df.duplicated().sum() / df.shape[0]) * 100
        
        summary_text = f"""
        This dataset contains {df.shape[0]:,} records and {df.shape[1]} variables. 
        Data completeness is {100 - missing_pct:.1f}% with {missing_pct:.1f}% missing values. 
        {duplicate_pct:.1f}% of records are duplicates. The analysis identified key patterns 
        and relationships that provide valuable insights for decision making.
        """
        
        story.append(Paragraph("Data Quality Summary:", self.heading_style))
        story.append(Paragraph(summary_text, self.normal_style))
        
        return story
    
    def _create_data_overview(self, df: pd.DataFrame, results: Dict[str, Any]) -> List:
        """Create data overview section"""
        story = []
        
        story.append(Paragraph("Data Overview", self.title_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Column information
        story.append(Paragraph("Column Information", self.heading_style))
        
        column_data = [["Column Name", "Data Type", "Non-Null Count", "Missing %"]]
        
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            column_data.append([
                col,
                str(df[col].dtype),
                f"{df[col].count():,}",
                f"{missing_pct:.1f}%"
            ])
        
        # Limit to first 20 columns to avoid overly long tables
        if len(column_data) > 21:
            column_data = column_data[:21]
            column_data.append(["...", "...", "...", "..."])
        
        table = Table(column_data, colWidths=[1.5 * inch, 1 * inch, 1 * inch, 0.8 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        
        return story
    
    def _create_statistical_analysis(self, results: Dict[str, Any]) -> List:
        """Create statistical analysis section"""
        story = []
        
        story.append(Paragraph("Statistical Analysis", self.title_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Summary statistics for numeric columns
        if 'summary_stats' in results and 'numeric' in results['summary_stats']:
            story.append(Paragraph("Numeric Variables Summary", self.heading_style))
            
            numeric_stats = results['summary_stats']['numeric']
            if numeric_stats:
                # Create summary table for first few numeric columns
                cols = list(numeric_stats.keys())[:5]  # Limit to 5 columns
                
                stats_data = [["Statistic"] + cols]
                
                for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
                    row = [stat.title()]
                    for col in cols:
                        if stat in numeric_stats[col]:
                            row.append(f"{numeric_stats[col][stat]:.2f}")
                        else:
                            row.append("N/A")
                    stats_data.append(row)
                
                table = Table(stats_data, colWidths=[1 * inch] + [0.8 * inch] * len(cols))
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(table)
        
        # Correlation analysis
        if 'correlations' in results and 'strong_correlations' in results['correlations']:
            story.append(Spacer(1, 0.3 * inch))
            story.append(Paragraph("Strong Correlations", self.heading_style))
            
            strong_corrs = results['correlations']['strong_correlations']
            if strong_corrs:
                for corr in strong_corrs[:10]:  # Top 10 correlations
                    story.append(Paragraph(
                        f"• {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f}",
                        self.normal_style
                    ))
            else:
                story.append(Paragraph("No strong correlations found (threshold: 0.7)", self.normal_style))
        
        return story
    
    def _create_data_quality_section(self, results: Dict[str, Any]) -> List:
        """Create data quality analysis section"""
        story = []
        
        story.append(Paragraph("Data Quality Analysis", self.title_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Missing values analysis
        if 'missing_values' in results:
            story.append(Paragraph("Missing Values", self.heading_style))
            
            missing_data = []
            for col, count in results['missing_values']['count'].items():
                if count > 0:
                    pct = results['missing_values']['percentage'][col]
                    missing_data.append([col, f"{count:,}", f"{pct:.1f}%"])
            
            if missing_data:
                missing_table_data = [["Column", "Missing Count", "Missing %"]] + missing_data[:15]
                table = Table(missing_table_data, colWidths=[2 * inch, 1 * inch, 1 * inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
            else:
                story.append(Paragraph("No missing values detected.", self.normal_style))
        
        # Outliers
        if 'outliers' in results:
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("Outlier Detection", self.heading_style))
            
            outlier_summary = []
            for col, outlier_info in results['outliers'].items():
                if outlier_info['count'] > 0:
                    outlier_summary.append(f"• {col}: {outlier_info['count']} outliers ({outlier_info['percentage']:.1f}%)")
            
            if outlier_summary:
                for summary in outlier_summary[:10]:
                    story.append(Paragraph(summary, self.normal_style))
            else:
                story.append(Paragraph("No significant outliers detected.", self.normal_style))
        
        return story
    
    def _create_insights_section(self, results: Dict[str, Any]) -> List:
        """Create insights and recommendations section"""
        story = []
        
        story.append(Paragraph("Insights & Recommendations", self.title_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Key insights
        if 'insights' in results:
            story.append(Paragraph("Key Insights", self.heading_style))
            for insight in results['insights']:
                story.append(Paragraph(f"• {insight}", self.normal_style))
        
        # Recommendations
        if 'recommendations' in results:
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("Recommendations", self.heading_style))
            for rec in results['recommendations']:
                story.append(Paragraph(f"• {rec}", self.normal_style))
        
        # Next steps
        if 'next_steps' in results:
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("Suggested Next Steps", self.heading_style))
            for step in results['next_steps']:
                story.append(Paragraph(f"• {step}", self.normal_style))
        
        return story
    
    def _create_visualizations_section(self, visualization_paths: List[str]) -> List:
        """Add visualizations to the report"""
        story = []
        
        story.append(Paragraph("Data Visualizations", self.title_style))
        story.append(Spacer(1, 0.3 * inch))
        
        for viz_path in visualization_paths:
            if os.path.exists(viz_path):
                try:
                    # Add chart title based on filename
                    chart_name = os.path.basename(viz_path).replace('.png', '').replace('_', ' ').title()
                    story.append(Paragraph(chart_name, self.heading_style))
                    
                    # Add image
                    img = Image(viz_path, width=6 * inch, height=4 * inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3 * inch))
                    
                except Exception as e:
                    logger.warning(f"Could not add visualization {viz_path}: {str(e)}")
        
        return story