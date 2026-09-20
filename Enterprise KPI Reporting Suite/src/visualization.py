"""
Visualization and Dashboard Module for Enterprise KPI Reporting Suite
Generates insight-driven visualizations and interactive Streamlit dashboard.

Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates visualizations and reports for enterprise KPI data.
    
    Creates executive-ready charts including:
    - Revenue vs Cost trends
    - Departmental target achievement heatmaps
    - LTV to CAC ratio distributions
    - Regional performance comparisons
    
    Attributes:
        df (pd.DataFrame): The input dataframe
        output_dir (Path): Directory for saving reports
    """
    
    # Class-level constant for author metadata
    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
    
    def __init__(self, df: pd.DataFrame, output_dir: str = 'reports'):
        """
        Initialize the Report Generator.
        
        Args:
            df: DataFrame containing KPI data
            output_dir: Directory path for saving generated reports
        """
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures: Dict[str, go.Figure] = {}
    
    def create_revenue_vs_cost_trend(self) -> go.Figure:
        """
        Create a time series chart showing Revenue vs Cost trends.
        
        Returns:
            go.Figure: Plotly figure with revenue and cost trends
        """
        # Aggregate by date
        daily_trends = self.df.groupby('Date').agg({
            'Revenue': 'sum',
            'Cost': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_trends['Date'],
            y=daily_trends['Revenue'],
            mode='lines',
            name='Revenue',
            line=dict(color='#2E86AB', width=2),
            fill='tozeroy',
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatter(
            x=daily_trends['Date'],
            y=daily_trends['Cost'],
            mode='lines',
            name='Cost',
            line=dict(color='#A23B72', width=2),
            fill='tozeroy',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Revenue vs Cost Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Amount ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(x=0, y=1.1, orientation='h')
        )
        
        self.figures['revenue_vs_cost_trend'] = fig
        logger.info("Created Revenue vs Cost trend chart")
        
        return fig
    
    def create_department_heatmap(self) -> go.Figure:
        """
        Create a heatmap showing departmental target achievement.
        
        Returns:
            go.Figure: Plotly heatmap figure
        """
        # Calculate achievement metrics by department
        dept_metrics = self.df.groupby('Department').agg({
            'Margin_Pct': 'mean',
            'LTV_CAC_Ratio': 'mean',
            'Conv_Rate_Pct': 'mean',
            'Attrition_Pct': 'mean',
            'Productivity_Pct': 'mean',
            'Training_Hrs': 'mean',
            'SLA_Met': 'mean',
            'Uptime_Pct': 'mean',
            'Resolution_Hrs': 'mean'
        }).round(2).T
        
        # Normalize values for better visualization
        normalized = dept_metrics.copy()
        for col in normalized.columns:
            col_data = normalized[col]
            col_min = col_data.min()
            col_max = col_data.max()
            if col_max > col_min:
                normalized[col] = (col_data - col_min) / (col_max - col_min) * 100
            else:
                normalized[col] = 50
        
        fig = go.Figure(data=go.Heatmap(
            z=normalized.values,
            x=normalized.columns,
            y=normalized.index,
            colorscale='RdYlGn',
            text=dept_metrics.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Departmental Target Achievement Heatmap',
            xaxis_title='Department',
            yaxis_title='Metric',
            height=400,
            template='plotly_white'
        )
        
        self.figures['department_heatmap'] = fig
        logger.info("Created departmental heatmap")
        
        return fig
    
    def create_ltv_cac_distribution(self) -> go.Figure:
        """
        Create a distribution chart for LTV to CAC ratio by department.
        
        Returns:
            go.Figure: Plotly box plot figure
        """
        fig = px.box(
            self.df,
            x='Department',
            y='LTV_CAC_Ratio',
            color='Department',
            title='LTV to CAC Ratio Distribution by Department',
            labels={'LTV_CAC_Ratio': 'LTV/CAC Ratio', 'Department': 'Department'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        # Add target line at 3.0
        fig.add_hline(
            y=3.0,
            line_dash="dash",
            line_color="red",
            annotation_text="Target: 3.0",
            annotation_position="top"
        )
        
        fig.update_layout(
            height=500,
            template='plotly_white',
            showlegend=False
        )
        
        self.figures['ltv_cac_distribution'] = fig
        logger.info("Created LTV/CAC distribution chart")
        
        return fig
    
    def create_regional_performance_chart(self) -> go.Figure:
        """
        Create a bar chart comparing regional performance.
        
        Returns:
            go.Figure: Plotly bar chart figure
        """
        regional_perf = self.df.groupby('Region').agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Deals_Closed': 'sum',
            'Margin_Pct': 'mean'
        }).reset_index()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Revenue by Region', 'Margin % by Region'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        fig.add_trace(
            go.Bar(
                x=regional_perf['Region'],
                y=regional_perf['Revenue'],
                name='Revenue',
                marker_color='#2E86AB'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=regional_perf['Region'],
                y=regional_perf['Margin_Pct'],
                name='Margin %',
                marker_color='#F18F01'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=450,
            template='plotly_white',
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Region", row=1, col=1)
        fig.update_xaxes(title_text="Region", row=1, col=2)
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
        fig.update_yaxes(title_text="Margin %", row=1, col=2)
        
        self.figures['regional_performance'] = fig
        logger.info("Created regional performance chart")
        
        return fig
    
    def create_quarterly_trend_chart(self) -> go.Figure:
        """
        Create a line chart showing quarterly revenue trends.
        
        Returns:
            go.Figure: Plotly line chart figure
        """
        quarterly = self.df.groupby(['Year', 'Quarter']).agg({
            'Revenue': 'sum',
            'Cost': 'sum'
        }).reset_index()
        quarterly['Period'] = quarterly['Year'].astype(str) + ' ' + quarterly['Quarter']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=quarterly['Period'],
            y=quarterly['Revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=quarterly['Period'],
            y=quarterly['Cost'],
            mode='lines+markers',
            name='Cost',
            line=dict(color='#A23B72', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='Quarterly Revenue and Cost Trends',
            xaxis_title='Period',
            yaxis_title='Amount ($)',
            hovermode='x unified',
            template='plotly_white',
            height=450,
            legend=dict(x=0, y=1.1, orientation='h')
        )
        
        self.figures['quarterly_trend'] = fig
        logger.info("Created quarterly trend chart")
        
        return fig
    
    def create_status_breakdown_pie(self) -> go.Figure:
        """
        Create a pie chart showing Above/Below Target breakdown.
        
        Returns:
            go.Figure: Plotly pie chart figure
        """
        status_counts = self.df['Status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        
        fig = px.pie(
            status_counts,
            values='Count',
            names='Status',
            title='Records Status Distribution',
            color='Status',
            color_discrete_map={
                'Above Target': '#2ECC71',
                'Below Target': '#E74C3C'
            }
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=450,
            template='plotly_white'
        )
        
        self.figures['status_breakdown'] = fig
        logger.info("Created status breakdown pie chart")
        
        return fig
    
    def save_all_figures(self, prefix: str = 'kpi_report') -> List[str]:
        """
        Save all generated figures as HTML files.
        
        Args:
            prefix: Prefix for saved file names
            
        Returns:
            List[str]: List of saved file paths
        """
        saved_files = []
        
        for name, fig in self.figures.items():
            file_path = self.output_dir / f"{prefix}_{name}.html"
            fig.write_html(str(file_path))
            saved_files.append(str(file_path))
            logger.info(f"Saved figure: {file_path}")
        
        return saved_files
    
    def generate_json_metrics(self, summary_data: Dict) -> str:
        """
        Generate a JSON file with key metrics and metadata.
        
        Args:
            summary_data: Dictionary containing executive summary data
            
        Returns:
            str: Path to the saved JSON file
        """
        # Ensure author metadata is included
        summary_data['author'] = self.AUTHOR
        summary_data['report_generated_at'] = pd.Timestamp.now().isoformat()
        
        file_path = self.output_dir / 'executive_metrics.json'
        
        with open(file_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        logger.info(f"Saved metrics JSON: {file_path}")
        
        return str(file_path)
    
    def create_executive_dashboard_data(self) -> Dict:
        """
        Prepare data structures for the executive dashboard.
        
        Returns:
            Dict: Dashboard data package
        """
        return {
            'author': self.AUTHOR,
            'total_records': len(self.df),
            'total_revenue': self.df['Revenue'].sum(),
            'total_cost': self.df['Cost'].sum(),
            'departments': list(self.df['Department'].unique()),
            'regions': list(self.df['Region'].unique()),
            'figures_available': list(self.figures.keys())
        }
