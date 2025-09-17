import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        
    def generate_data_profile(self):
        """Generate a comprehensive data quality report"""
        profile = {}
        
        # Basic overview
        profile['overview'] = pd.DataFrame({
            'Column': self.df.columns,
            'Non-Null Count': self.df.count().values,
            'Null Count': self.df.isnull().sum().values,
            'Data Type': self.df.dtypes.values
        })
        
        # Missing values
        missing_values = self.df.isnull().sum()
        missing_percent = (missing_values / len(self.df)) * 100
        profile['missing_values'] = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Values': missing_values.values,
            'Percentage': missing_percent.values
        })
        
        # Data types
        profile['data_types'] = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': self.df.dtypes.values
        })
        
        # Descriptive statistics (numeric only)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            profile['stats'] = self.df[numeric_cols].describe().T
        else:
            profile['stats'] = pd.DataFrame()
        
        # Identify data quality issues (optimized)
        profile['issues'] = self._identify_data_issues_fast()
        
        return profile
    
    def _identify_data_issues_fast(self):
        """Fast data quality issue detection"""
        issues = []
        
        # Fast checks only
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if missing_cols:
            issues.append(f"Missing values detected in columns: {', '.join(missing_cols)}")
        
        duplicate_rows = self.df.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(f"{duplicate_rows} duplicate rows found in the dataset")
        
        return issues
    
    def auto_clean_data(self):
        """Automatically clean the dataset"""
        cleaned_df = self.df.copy()
        
        # Handle missing values
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().sum() > 0:
                if cleaned_df[col].dtype == 'object' or cleaned_df[col].dtype.name == 'category':
                    # For categorical, fill with mode
                    mode_val = cleaned_df[col].mode()
                    if not mode_val.empty:
                        cleaned_df[col].fillna(mode_val[0], inplace=True)
                    else:
                        cleaned_df[col].fillna('Unknown', inplace=True)
                elif pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    # For numerical, fill with median
                    median_val = cleaned_df[col].median()
                    cleaned_df[col].fillna(median_val, inplace=True)
        
        # Remove duplicates
        cleaned_df.drop_duplicates(inplace=True)
        
        return cleaned_df