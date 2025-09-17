import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import random

class ChartGenerator:
    def __init__(self, df):
        self.df = df
        # Define multiple color palettes for variety
        self.color_palettes = [
            px.colors.qualitative.Set3,
            px.colors.qualitative.Pastel,
            px.colors.qualitative.Bold,
            px.colors.qualitative.Vivid,
            px.colors.qualitative.Safe
        ]
    
    def generate_all_charts(self):
        """Generate diverse chart types for different data aspects"""
        charts = []
        
        # Get column types
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # 1. For numeric columns: Histograms or Box plots
        for col in numeric_cols[:3]:  # Limit to 3 numeric charts
            if self.df[col].nunique() > 1:
                # Alternate between histogram and box plot
                if len(charts) % 2 == 0:
                    fig = self._create_histogram(col)
                else:
                    fig = self._create_box_plot(col)
                charts.append((fig, "distribution", [col]))
        
        # 2. For categorical columns: Diverse chart types
        for i, col in enumerate(categorical_cols[:4]):  # Limit to 4 categorical charts
            if 2 <= self.df[col].nunique() <= 15:
                # Rotate through different chart types
                chart_type = i % 4  # 0: bar, 1: pie, 2: donut, 3: treemap
                
                if chart_type == 0:
                    fig = self._create_bar_chart(col)
                elif chart_type == 1:
                    fig = self._create_pie_chart(col)
                elif chart_type == 2:
                    fig = self._create_donut_chart(col)
                else:
                    fig = self._create_treemap(col)
                
                charts.append((fig, "categorical", [col]))
        
        # 3. Correlation heatmap (if enough numeric columns)
        if len(numeric_cols) > 1:
            fig = self._create_correlation_heatmap(numeric_cols)
            charts.append((fig, "correlation", numeric_cols))
        
        # 4. Scatter plot for numeric relationships
        if len(numeric_cols) >= 2:
            fig = self._create_scatter_plot(numeric_cols[0], numeric_cols[1], categorical_cols)
            charts.append((fig, "relationship", [numeric_cols[0], numeric_cols[1]]))
        
        # 5. Time series line chart (if datetime available)
        if datetime_cols and numeric_cols:
            fig = self._create_time_series(datetime_cols[0], numeric_cols[0])
            charts.append((fig, "timeseries", [datetime_cols[0], numeric_cols[0]]))
        
        # 6. Violin plot for distribution comparison (if categorical and numeric)
        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            if self.df[cat_col].nunique() <= 6:
                fig = self._create_violin_plot(cat_col, num_col)
                charts.append((fig, "comparison", [cat_col, num_col]))
        
        return charts
    
    def _get_random_palette(self):
        """Get a random color palette"""
        return random.choice(self.color_palettes)
    
    def _create_histogram(self, col):
        """Create a histogram with random colors"""
        palette = self._get_random_palette()
        fig = px.histogram(
            self.df, 
            x=col, 
            title=f"Distribution of {col}",
            color_discrete_sequence=palette,
            nbins=30,
            opacity=0.8
        )
        fig.update_layout(
            bargap=0.1,
            plot_bgcolor='rgba(240,240,240,0.1)'
        )
        return fig
    
    def _create_box_plot(self, col):
        """Create a box plot"""
        palette = self._get_random_palette()
        fig = px.box(
            self.df,
            y=col,
            title=f"Box Plot of {col}",
            color_discrete_sequence=palette
        )
        return fig
    
    def _create_bar_chart(self, col):
        """Create a colorful bar chart"""
        value_counts = self.df[col].value_counts().reset_index()
        value_counts.columns = [col, 'count']
        
        palette = self._get_random_palette()
        fig = px.bar(
            value_counts, 
            x=col, 
            y='count', 
            title=f"{col} Distribution",
            color=col,
            color_discrete_sequence=palette
        )
        fig.update_layout(showlegend=False)
        return fig
    
    def _create_pie_chart(self, col):
        """Create a pie chart"""
        value_counts = self.df[col].value_counts().reset_index()
        value_counts.columns = [col, 'count']
        
        palette = self._get_random_palette()
        fig = px.pie(
            value_counts,
            names=col,
            values='count',
            title=f"{col} Distribution",
            color_discrete_sequence=palette
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig
    
    def _create_donut_chart(self, col):
        """Create a donut chart"""
        value_counts = self.df[col].value_counts().reset_index()
        value_counts.columns = [col, 'count']
        
        palette = self._get_random_palette()
        fig = go.Figure(data=[go.Pie(
            labels=value_counts[col],
            values=value_counts['count'],
            hole=0.5,
            marker_colors=palette,
            textinfo='label+percent',
            textposition='inside'
        )])
        fig.update_layout(title_text=f"Donut Chart: {col}")
        return fig
    
    def _create_treemap(self, col):
        """Create a treemap chart"""
        value_counts = self.df[col].value_counts().reset_index()
        value_counts.columns = [col, 'count']
        
        palette = self._get_random_palette()
        fig = px.treemap(
            value_counts,
            path=[col],
            values='count',
            title=f"Treemap: {col}",
            color='count',
            color_continuous_scale=palette
        )
        return fig
    
    def _create_correlation_heatmap(self, numeric_cols):
        """Create correlation heatmap"""
        corr_matrix = self.df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix, 
            title="Correlation Matrix",
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(coloraxis_colorbar=dict(title="Correlation"))
        return fig
    
    def _create_scatter_plot(self, col1, col2, categorical_cols):
        """Create scatter plot with optional color coding"""
        color_col = None
        if categorical_cols and self.df[categorical_cols[0]].nunique() <= 8:
            color_col = categorical_cols[0]
            palette = self._get_random_palette()
        else:
            palette = None
        
        fig = px.scatter(
            self.df, 
            x=col1, 
            y=col2, 
            title=f"{col1} vs {col2}",
            color=color_col,
            color_discrete_sequence=palette
        )
        return fig
    
    def _create_time_series(self, date_col, num_col):
        """Create time series line chart"""
        time_df = self.df.groupby(date_col)[num_col].mean().reset_index()
        
        palette = self._get_random_palette()
        fig = px.line(
            time_df, 
            x=date_col, 
            y=num_col, 
            title=f" {num_col} over Time",
            color_discrete_sequence=palette
        )
        fig.update_traces(line=dict(width=3))
        return fig
    
    def _create_violin_plot(self, cat_col, num_col):
        """Create violin plot"""
        palette = self._get_random_palette()
        fig = px.violin(
            self.df,
            x=cat_col,
            y=num_col,
            title=f"{num_col} by {cat_col}",
            color=cat_col,
            color_discrete_sequence=palette,
            box=True
        )
        return fig