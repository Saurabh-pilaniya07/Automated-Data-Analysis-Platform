import pandas as pd
import numpy as np
import plotly.express as px
import re
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

class QASystem:
    def __init__(self, df):
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Initialize HuggingFace QA model
        self.qa_pipeline = None
        self._initialize_qa_model()
    
    def _initialize_qa_model(self):
        """Initialize the HuggingFace QA model"""
        try:
            # Use a lightweight model that's good for QA
            model_name = "distilbert-base-cased-distilled-squad"
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                tokenizer=model_name,
                device=-1,  # Use CPU (-1) instead of GPU
                torch_dtype='auto'
            )
            print("HuggingFace QA model loaded successfully!")
        except Exception as e:
            print(f"Failed to load HuggingFace model: {e}")
            print("Falling back to rule-based approach only")
            self.qa_pipeline = None
    
    def answer_question(self, question):
        """Answer natural language questions about the data"""
        try:
            question_lower = question.lower()
            
            # DIRECT all statistical questions to rule-based to avoid AI confusion
            statistical_keywords = ['average', 'mean', 'median', 'sum', 'total', 'count', 
                                   'minimum', 'maximum', 'min', 'max', 'std', 'deviation',
                                   'how many', 'number of']
            
            if any(keyword in question_lower for keyword in statistical_keywords):
                return self._rule_based_qa(question)
            
            # First try with HuggingFace model if available (for complex questions)
            if self.qa_pipeline:
                # Convert dataframe to context text for the model
                context = self._dataframe_to_context()
                if context and len(context) > 100:  # Ensure we have enough context
                    result = self.qa_pipeline(question=question, context=context)
                    if result['score'] > 0.3:  # Reasonable confidence threshold
                        answer = f"{result['answer']} (confidence: {result['score']:.2f})"
                        chart = self._generate_chart_from_question(question)
                        return answer, chart
            
            # Fallback to rule-based approach
            return self._rule_based_qa(question)
            
        except Exception as e:
            return f"I encountered an error processing your question: {str(e)}", None
    
    def _dataframe_to_context(self):
        """Convert dataframe to text context for QA model - FIXED"""
        try:
            context = "Dataset Summary:\n"
            context += f"- Total records: {len(self.df)}\n"
            context += f"- Total columns: {len(self.df.columns)}\n"
            context += f"- Numeric columns: {len(self.numeric_cols)}\n"
            context += f"- Categorical columns: {len(self.categorical_cols)}\n"
            context += f"- Date/time columns: {len(self.datetime_cols)}\n\n"
            
            context += "Column Information (Statistics Only - No Raw Data):\n"
            for col in self.df.columns:
                col_type = str(self.df[col].dtype)
                non_null = self.df[col].count()
                unique_vals = self.df[col].nunique()
                null_count = self.df[col].isnull().sum()
                null_percent = (null_count / len(self.df)) * 100
                
                context += f"- {col} ({col_type}): {non_null} non-null, {null_count} null ({null_percent:.1f}%), {unique_vals} unique values"
                
                # Add statistics for numeric columns (NOT raw data)
                if col in self.numeric_cols:
                    context += f", min={self.df[col].min():.2f}, max={self.df[col].max():.2f}, mean={self.df[col].mean():.2f}"
                context += "\n"
            
            # Add column descriptions without actual data
            context += "\nColumn Descriptions:\n"
            for col in self.df.columns:
                if col in self.numeric_cols:
                    context += f"- {col}: Numeric measurement ranging from {self.df[col].min():.2f} to {self.df[col].max():.2f}\n"
                elif col in self.categorical_cols:
                    top_values = self.df[col].value_counts().head(3)
                    context += f"- {col}: Categorical variable with {self.df[col].nunique()} categories. Most common: {', '.join([str(k) for k in top_values.index[:2]])}\n"
                else:
                    context += f"- {col}: {col_type} data type\n"
            
            return context
            
        except Exception as e:
            print(f"Error creating context: {e}")
            return None
        
    def _rule_based_qa(self, question):
        """Simple rule-based question answering as fallback"""
        question_lower = question.lower()
        
        # Which has the highest
        if any(word in question_lower for word in ['highest', 'maximum', 'max', 'largest', 'biggest']):
            return self._answer_extremum(question, "max")
        
        # Which has the lowest
        elif any(word in question_lower for word in ['lowest', 'minimum', 'min', 'smallest', 'least']):
            return self._answer_extremum(question, "min")
        
        # What is the average
        elif any(word in question_lower for word in ['average', 'mean', 'avg']):
            return self._answer_statistical(question, "mean")
        
        # Median questions
        elif 'median' in question_lower:
            return self._answer_statistical(question, "median")
        
        # Count questions
        elif any(word in question_lower for word in ['how many', 'count', 'number of', 'total']):
            return self._answer_count(question)
        
        # Correlation/relationship questions
        elif any(word in question_lower for word in ['correlation', 'relationship', 'related', 'connect']):
            return self._answer_correlation(question)
        
        # Distribution questions
        elif any(word in question_lower for word in ['distribution', 'histogram', 'frequency', 'spread']):
            return self._answer_distribution(question)
        
        # Trend over time questions
        elif any(word in question_lower for word in ['trend', 'over time', 'time series', 'across time']):
            return self._answer_trend(question)
        
        # Unique values questions
        elif any(word in question_lower for word in ['unique', 'different', 'distinct']):
            return self._answer_unique_values(question)
        
        else:
            return self._provide_general_help(), None
    
    def _answer_extremum(self, question, ext_type):
        """Answer questions about maximum or minimum values"""
        question_lower = question.lower()
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        # Find likely column reference
        target_col = None
        for col in self.df.columns:
            if col.lower() in question_lower:
                if col in numeric_cols:
                    target_col = col
                    break
        
        if not target_col and len(numeric_cols) > 0:
            target_col = numeric_cols[0]  # Default to first numeric column
        
        if target_col:
            if ext_type == "max":
                value = self.df[target_col].max()
                idx = self.df[target_col].idxmax()
                result_row = self.df.iloc[idx]
                
                # Try to find a categorical column to show which category has the max
                cat_cols = self.df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    category_col = cat_cols[0]
                    category_val = result_row[category_col]
                    answer = f"The maximum {target_col} is {value:.2f}, which occurs for {category_col} = {category_val}."
                    
                    # Create a bar chart of top values
                    top_values = self.df.nlargest(10, target_col)
                    fig = px.bar(top_values, x=category_col, y=target_col, 
                                title=f"Top 10 {target_col} values by {category_col}")
                    return answer, fig
                else:
                    answer = f"The maximum {target_col} is {value:.2f}."
                    return answer, None
            else:  # min
                value = self.df[target_col].min()
                idx = self.df[target_col].idxmin()
                result_row = self.df.iloc[idx]
                
                cat_cols = self.df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    category_col = cat_cols[0]
                    category_val = result_row[category_col]
                    answer = f"The minimum {target_col} is {value:.2f}, which occurs for {category_col} = {category_val}."
                    
                    # Create a bar chart of bottom values
                    bottom_values = self.df.nsmallest(10, target_col)
                    fig = px.bar(bottom_values, x=category_col, y=target_col, 
                                title=f"Bottom 10 {target_col} values by {category_col}")
                    return answer, fig
                else:
                    answer = f"The minimum {target_col} is {value:.2f}."
                    return answer, None
        else:
            return "I couldn't identify a numeric column in your question. Please specify which column you're interested in.", None
    
    def _answer_statistical(self, question, stat_type):
        """Answer statistical questions"""
        question_lower = question.lower()
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        # Find likely column reference
        target_col = None
        for col in self.df.columns:
            if col.lower() in question_lower:
                if col in numeric_cols:
                    target_col = col
                    break
        
        if not target_col and len(numeric_cols) > 0:
            target_col = numeric_cols[0]  # Default to first numeric column
        
        if target_col:
            if stat_type == "mean":
                value = self.df[target_col].mean()
                answer = f"The average {target_col} is {value:.2f}."
                return answer, None
            elif stat_type == "median":
                value = self.df[target_col].median()
                answer = f"The median {target_col} is {value:.2f}."
                return answer, None
        else:
            return "I couldn't identify a numeric column in your question. Please specify which column you're interested in.", None
    
    def _answer_count(self, question):
        """Answer count questions"""
        question_lower = question.lower()
        
        # Count total records
        if any(word in question_lower for word in ['record', 'row', 'total', 'dataset', 'how many']):
            count = len(self.df)
            return f"There are {count} total records in the dataset.", None
        
        # Count by category
        target_col = self._find_column_in_question(question_lower, self.categorical_cols)
        if target_col:
            value_counts = self.df[target_col].value_counts()
            if len(value_counts) > 10:
                top_5 = value_counts.head(5)
                answer = f"Top 5 categories in {target_col}:\n"
                for category, count in top_5.items():
                    percentage = (count / len(self.df)) * 100
                    answer += f"- {category}: {count} records ({percentage:.1f}%)\n"
                return answer, None
            else:
                answer = f"Value counts for {target_col}:\n"
                for category, count in value_counts.items():
                    percentage = (count / len(self.df)) * 100
                    answer += f"- {category}: {count} records ({percentage:.1f}%)\n"
                return answer, None
        
        return "I couldn't understand what you want to count. Try specifying a category column.", None
    
    def _answer_correlation(self, question):
        """Answer correlation questions"""
        question_lower = question.lower()
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            correlation = self.df[col1].corr(self.df[col2])
            
            answer = f"The correlation between {col1} and {col2} is {correlation:.2f}. "
            if correlation > 0.7:
                answer += "This indicates a strong positive relationship."
            elif correlation > 0.3:
                answer += "This indicates a moderate positive relationship."
            elif correlation > -0.3:
                answer += "This indicates a weak relationship."
            elif correlation > -0.7:
                answer += "This indicates a moderate negative relationship."
            else:
                answer += "This indicates a strong negative relationship."
            
            # Create a scatter plot
            fig = px.scatter(self.df, x=col1, y=col2, title=f"Relationship between {col1} and {col2}")
            return answer, fig
        else:
            return "I need at least two numeric columns to calculate a correlation.", None
    
    def _answer_distribution(self, question):
        """Answer distribution questions"""
        target_col = self._find_column_in_question(question, self.numeric_cols)
        
        if not target_col and self.numeric_cols:
            target_col = self.numeric_cols[0]
        
        if target_col:
            stats = self.df[target_col].describe()
            answer = f"Distribution of {target_col}:\n"
            answer += f"- Mean: {stats['mean']:.2f}\n"
            answer += f"- Median: {self.df[target_col].median():.2f}\n"
            answer += f"- Standard Deviation: {stats['std']:.2f}\n"
            answer += f"- Range: {stats['min']:.2f} to {stats['max']:.2f}\n"
            answer += f"- 25th percentile: {stats['25%']:.2f}\n"
            answer += f"- 75th percentile: {stats['75%']:.2f}"
            
            # Create histogram
            chart = px.histogram(self.df, x=target_col, title=f"Distribution of {target_col}")
            return answer, chart
        
        return "I couldn't identify a numeric column for distribution analysis.", None
    
    def _answer_trend(self, question):
        """Answer trend over time questions"""
        if self.datetime_cols and self.numeric_cols:
            date_col = self.datetime_cols[0]
            num_col = self.numeric_cols[0]
            
            # Create time series
            time_df = self.df.groupby(date_col)[num_col].mean().reset_index()
            
            answer = f"Trend analysis of {num_col} over {date_col}: "
            answer += f"From {time_df[date_col].min()} to {time_df[date_col].max()}, "
            answer += f"showing {len(time_df)} time points."
            
            # Create line chart
            chart = px.line(time_df, x=date_col, y=num_col, title=f"{num_col} over Time")
            return answer, chart
        
        return "I need both date/time and numeric columns to analyze trends.", None
    
    def _answer_unique_values(self, question):
        """Answer questions about unique values"""
        target_col = self._find_column_in_question(question, self.df.columns)
        
        if target_col:
            unique_count = self.df[target_col].nunique()
            answer = f"Column '{target_col}' has {unique_count} unique values."
            
            if unique_count <= 10:
                unique_values = self.df[target_col].unique()
                answer += f"\nUnique values: {', '.join(map(str, unique_values))}"
            
            return answer, None
        
        return "Please specify which column you want to check for unique values.", None
    
    def _find_column_in_question(self, question_lower, possible_columns):
        """Find which column is mentioned in the question"""
        for col in possible_columns:
            if col.lower() in question_lower:
                return col
        return None
    
    def _generate_chart_from_question(self, question):
        """Try to generate an appropriate chart based on the question"""
        question_lower = question.lower()
        
        # Check for comparison terms
        comparison_terms = ["compare", "versus", "vs", "difference", "relationship"]
        if any(term in question_lower for term in comparison_terms):
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            if len(numeric_cols) >= 2:
                return px.scatter(self.df, x=numeric_cols[0], y=numeric_cols[1], 
                                 title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
        
        # Check for distribution terms
        distribution_terms = ["distribution", "histogram', 'frequency"]
        if any(term in question_lower for term in distribution_terms):
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                return px.histogram(self.df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}")
        
        # Check for trend terms
        trend_terms = ["trend", "over time", "time series"]
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        if any(term in question_lower for term in trend_terms) and datetime_cols and numeric_cols:
            time_series = self.df.groupby(datetime_cols[0])[numeric_cols[0]].mean().reset_index()
            return px.line(time_series, x=datetime_cols[0], y=numeric_cols[0], 
                          title=f"{numeric_cols[0]} over Time")
        
        return None
    
    def _provide_general_help(self):
        """Provide general help when question isn't understood"""
        help_text = "I can answer questions about:\n"
        help_text += "• **Maximum/minimum values**: 'Which branch has the highest sales?'\n"
        help_text += "• **Averages**: 'What is the average age?'\n"
        help_text += "• **Counts**: 'How many customers in each category?'\n"
        help_text += "• **Relationships**: 'What is the correlation between income and spending?'\n"
        help_text += "• **Distributions**: 'Show me the distribution of prices'\n"
        help_text += "• **Trends**: 'What is the trend over time?'\n"
        help_text += "• **Unique values**: 'How many unique products do we have?'\n\n"
        help_text += "**Available numeric columns**: " + ", ".join(self.numeric_cols[:5]) + ("..." if len(self.numeric_cols) > 5 else "") + "\n"
        help_text += "**Available categorical columns**: " + ", ".join(self.categorical_cols[:5]) + ("..." if len(self.categorical_cols) > 5 else "")
        
        return help_text