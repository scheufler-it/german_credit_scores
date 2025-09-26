import pandas as pd
import numpy as np
from collections import Counter
import json

class GermanCreditCleanser:
    def __init__(self, file_path='german_credit_dirty.csv'):
        self.file_path = file_path
        self.original_data = None
        self.cleaned_data = None
        self.cleaning_report = {}
        
    def load_data(self):
        """Load the CSV file"""
        print("Loading dataset...")
        try:
            self.original_data = pd.read_csv(self.file_path)
            print(f"✓ Successfully loaded {len(self.original_data)} rows and {len(self.original_data.columns)} columns")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def analyze_data(self):
        """Analyze the dataset for quality issues"""
        if self.original_data is None:
            print("No data loaded. Please load data first.")
            return None
            
        print("\n" + "="*50)
        print("DATA ANALYSIS")
        print("="*50)
        
        analysis = {
            'total_rows': len(self.original_data),
            'total_columns': len(self.original_data.columns),
            'duplicates': 0,
            'missing_values': {},
            'invalid_values': {},
            'outliers': {},
            'data_types': {}
        }
        
        # Check for duplicates
        duplicates = self.original_data.duplicated().sum()
        analysis['duplicates'] = duplicates
        print(f"Total rows: {analysis['total_rows']}")
        print(f"Duplicate rows: {duplicates}")
        
        # Analyze each column
        print(f"\nColumn Analysis:")
        print("-" * 30)
        
        for col in self.original_data.columns:
            # Missing values
            missing = self.original_data[col].isnull().sum()
            empty_strings = (self.original_data[col] == '').sum()
            null_strings = self.original_data[col].isin(['NULL', 'null']).sum()
            total_missing = missing + empty_strings + null_strings
            
            if total_missing > 0:
                analysis['missing_values'][col] = total_missing
            
            # Data type analysis
            analysis['data_types'][col] = str(self.original_data[col].dtype)
            
            # Check for invalid values in specific columns
            if col in ['age', 'loan_amt']:
                # Check for non-positive numeric values
                try:
                    numeric_col = pd.to_numeric(self.original_data[col], errors='coerce')
                    invalid = ((numeric_col <= 0) | numeric_col.isnull()).sum() - total_missing
                    if invalid > 0:
                        analysis['invalid_values'][col] = invalid
                        
                    # Outlier detection for loan_amt
                    if col == 'loan_amt':
                        valid_loans = numeric_col[(numeric_col > 0) & numeric_col.notna()]
                        if len(valid_loans) > 0:
                            q1 = valid_loans.quantile(0.25)
                            q3 = valid_loans.quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            
                            # Typical German personal loans range from 500 to 100,000 EUR
                            reasonable_max = 100000  # 100k EUR
                            reasonable_min = 100     # 100 EUR
                            
                            final_upper = reasonable_max
                            final_lower = reasonable_min
                            
                            outliers = valid_loans[(valid_loans < final_lower) | (valid_loans > final_upper)]
                            
                            if len(outliers) > 0:
                                analysis['outliers'][col] = {
                                    'count': len(outliers),
                                    'min_outlier': outliers.min(),
                                    'max_outlier': outliers.max(),
                                    'reasonable_range': f"{final_lower:.0f} - {final_upper:.0f}",
                                    'q1': q1,
                                    'q3': q3,
                                    'iqr_upper': upper_bound,
                                    'final_upper': final_upper
                                }
                                
                            print(f"{col:25} | Type: {analysis['data_types'][col]:10} | Missing: {total_missing:3} | Range: {valid_loans.min():.0f}-{valid_loans.max():.0f}")
                        else:
                            print(f"{col:25} | Type: {analysis['data_types'][col]:10} | Missing: {total_missing:3}")
                    else:
                        # For age, also show range
                        valid_ages = numeric_col[(numeric_col > 0) & numeric_col.notna()]
                        if len(valid_ages) > 0:
                            print(f"{col:25} | Type: {analysis['data_types'][col]:10} | Missing: {total_missing:3} | Range: {valid_ages.min():.0f}-{valid_ages.max():.0f}")
                        else:
                            print(f"{col:25} | Type: {analysis['data_types'][col]:10} | Missing: {total_missing:3}")
                except:
                    print(f"{col:25} | Type: {analysis['data_types'][col]:10} | Missing: {total_missing:3}")
                    pass
            
            elif col in ['duration', 'installment_rate', 'present_residence_since', 
                        'num_curr_loans', 'num_people_provide_maint']:
                # Check for negative integer values
                try:
                    numeric_col = pd.to_numeric(self.original_data[col], errors='coerce')
                    invalid = ((numeric_col < 0) | numeric_col.isnull()).sum() - total_missing
                    if invalid > 0:
                        analysis['invalid_values'][col] = invalid
                    
                    valid_vals = numeric_col[numeric_col.notna() & (numeric_col >= 0)]
                    if len(valid_vals) > 0:
                        print(f"{col:25} | Type: {analysis['data_types'][col]:10} | Missing: {total_missing:3} | Range: {valid_vals.min():.0f}-{valid_vals.max():.0f}")
                    else:
                        print(f"{col:25} | Type: {analysis['data_types'][col]:10} | Missing: {total_missing:3}")
                except:
                    print(f"{col:25} | Type: {analysis['data_types'][col]:10} | Missing: {total_missing:3}")
            else:
                print(f"{col:25} | Type: {analysis['data_types'][col]:10} | Missing: {total_missing:3}")
        
        # Display missing values summary
        if analysis['missing_values']:
            print(f"\nColumns with missing values:")
            for col, count in analysis['missing_values'].items():
                print(f"  {col}: {count} missing values")
        else:
            print(f"\n✓ No missing values detected")
        
        # Display invalid values summary
        if analysis['invalid_values']:
            print(f"\nColumns with invalid values:")
            for col, count in analysis['invalid_values'].items():
                print(f"  {col}: {count} invalid values")
        else:
            print(f"✓ No invalid values detected")
        
        # Display outliers summary
        if analysis['outliers']:
            print(f"\nOutliers detected:")
            for col, info in analysis['outliers'].items():
                print(f"  {col}: {info['count']} outliers (range: {info['min_outlier']:.0f} - {info['max_outlier']:.0f})")
                print(f"    Reasonable range: {info['reasonable_range']}")
        else:
            print(f"✓ No significant outliers detected")
            
        return analysis
    
    def clean_data(self):
        """Clean the dataset"""
        if self.original_data is None:
            print("No data loaded. Please load data first.")
            return False
            
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)
        
        # Initialize cleaning report
        self.cleaning_report = {
            'original_count': len(self.original_data),
            'operations': [],
            'final_count': 0
        }
        
        # Start with a copy of original data
        cleaned = self.original_data.copy()
        
        print(f"Starting with {len(cleaned)} rows")
        
        # Step 1: Remove duplicates
        duplicates_before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        duplicates_removed = duplicates_before - len(cleaned)
        
        if duplicates_removed > 0:
            operation = f"Removed {duplicates_removed} duplicate rows"
            self.cleaning_report['operations'].append(operation)
            print(f"✓ {operation}")
        
        # Step 2: Handle missing values and invalid data in critical columns
        rows_before = len(cleaned)
        
        # Remove rows with missing target variable
        cleaned = cleaned[cleaned['target'].notna()]
        cleaned = cleaned[cleaned['target'] != '']
        cleaned = cleaned[~cleaned['target'].isin(['NULL', 'null'])]
        
        # Remove rows with missing or invalid age
        cleaned['age_numeric'] = pd.to_numeric(cleaned['age'], errors='coerce')
        cleaned = cleaned[cleaned['age_numeric'].notna()]
        cleaned = cleaned[cleaned['age_numeric'] > 0]
        
        # Remove rows with missing or invalid loan amount AND outliers
        cleaned['loan_amt_numeric'] = pd.to_numeric(cleaned['loan_amt'], errors='coerce')
        cleaned = cleaned[cleaned['loan_amt_numeric'].notna()]
        cleaned = cleaned[cleaned['loan_amt_numeric'] > 0]
        
        # Remove loan amount outliers
        valid_loans = cleaned['loan_amt_numeric']
        if len(valid_loans) > 0:
            
            # Set reasonable absolute bounds for German credit (typical range: 500-100,000 EUR)
            reasonable_max = 100000  # 100k EUR maximum
            reasonable_min = 100     # 100 EUR minimum
            
            
            final_upper = reasonable_max
            final_lower = reasonable_min
            
            outliers_before = len(cleaned)
            cleaned = cleaned[
                (cleaned['loan_amt_numeric'] >= final_lower) & 
                (cleaned['loan_amt_numeric'] <= final_upper)
            ]
            outliers_removed = outliers_before - len(cleaned)
            
            if outliers_removed > 0:
                operation = f"Removed {outliers_removed} loan amount outliers (outside range {final_lower:.0f}-{final_upper:.0f})"
                self.cleaning_report['operations'].append(operation)
                print(f"✓ {operation}")
        
        # Drop the temporary numeric columns
        cleaned = cleaned.drop(['age_numeric', 'loan_amt_numeric'], axis=1)
        
        invalid_removed = rows_before - len(cleaned)
        if invalid_removed > 0:
            operation = f"Removed {invalid_removed} rows with missing or invalid critical data (excluding outlier removal)"
            self.cleaning_report['operations'].append(operation)
            print(f"✓ {operation}")
        
        # Step 3: Clean and standardize data types
        print("✓ Standardizing data types and cleaning string values")
        
        # Clean numeric fields
        numeric_fields = ['age', 'loan_amt']
        for field in numeric_fields:
            cleaned[field] = pd.to_numeric(cleaned[field], errors='coerce')
            cleaned[field] = cleaned[field].astype(str)
        
        # Clean integer fields
        integer_fields = ['duration', 'installment_rate', 'present_residence_since', 
                         'num_curr_loans', 'num_people_provide_maint']
        for field in integer_fields:
            if field in cleaned.columns:
                # Convert to numeric, then to int, then to string
                numeric_vals = pd.to_numeric(cleaned[field], errors='coerce')
                # Fill NaN with the column's mode (most frequent value) or 0
                if numeric_vals.notna().sum() > 0:
                    mode_val = numeric_vals.mode().iloc[0] if len(numeric_vals.mode()) > 0 else 0
                    numeric_vals = numeric_vals.fillna(mode_val)
                    cleaned[field] = numeric_vals.astype(int)
        
        # Clean string fields
        string_fields = [col for col in cleaned.columns if col not in numeric_fields + integer_fields]
        for field in string_fields:
            if cleaned[field].dtype == 'object':
                # Remove extra whitespace and handle null values
                cleaned[field] = cleaned[field].astype(str).str.strip()
                cleaned[field] = cleaned[field].replace(['NULL', 'null', 'nan'], '')
        
        operation = "Standardized data types and cleaned string values"
        self.cleaning_report['operations'].append(operation)
        
        # Final report
        self.cleaning_report['final_count'] = len(cleaned)
        operation = f"Final dataset: {len(cleaned)} rows"
        self.cleaning_report['operations'].append(operation)
        print(f"✓ {operation}")
        
        self.cleaned_data = cleaned
        return True
    
    def display_cleaning_report(self):
        """Display the cleaning report"""
        if not self.cleaning_report:
            print("No cleaning operations performed yet.")
            return
            
        print("\n" + "="*50)
        print("CLEANING REPORT")
        print("="*50)
        
        print(f"Original dataset: {self.cleaning_report['original_count']} rows")
        print(f"Final dataset: {self.cleaning_report['final_count']} rows")
        print(f"Rows removed: {self.cleaning_report['original_count'] - self.cleaning_report['final_count']}")
        
        print(f"\nOperations performed:")
        for i, operation in enumerate(self.cleaning_report['operations'], 1):
            print(f"  {i}. {operation}")
    
    def save_cleaned_data(self, output_file='german_credit_cleaned.csv'):
        """Save the cleaned dataset"""
        if self.cleaned_data is None:
            print("No cleaned data available. Please run cleaning first.")
            return False
            
        try:
            self.cleaned_data.to_csv(output_file, index=False)
            print(f"\n✓ Cleaned dataset saved to '{output_file}'")
            print(f"  Final size: {len(self.cleaned_data)} rows × {len(self.cleaned_data.columns)} columns")
            return True
        except Exception as e:
            print(f"✗ Error saving file: {e}")
            return False
    
    def get_data_summary(self):
        """Get summary statistics of the cleaned data"""
        if self.cleaned_data is None:
            print("No cleaned data available.")
            return
            
        print("\n" + "="*50)
        print("CLEANED DATA SUMMARY")
        print("="*50)
        
        print(f"Dataset shape: {self.cleaned_data.shape}")
        print(f"\nColumn types:")
        for col in self.cleaned_data.columns:
            dtype = self.cleaned_data[col].dtype
            unique_vals = self.cleaned_data[col].nunique()
            
            # Show range for numeric columns
            if col in ['age', 'loan_amt']:
                try:
                    numeric_col = pd.to_numeric(self.cleaned_data[col], errors='coerce')
                    if numeric_col.notna().sum() > 0:
                        min_val = numeric_col.min()
                        max_val = numeric_col.max()
                        mean_val = numeric_col.mean()
                        print(f"  {col:25} | Type: {str(dtype):10} | Unique: {unique_vals:4} | Range: {min_val:.0f}-{max_val:.0f} (Mean: {mean_val:.0f})")
                    else:
                        print(f"  {col:25} | Type: {str(dtype):10} | Unique: {unique_vals:4}")
                except:
                    print(f"  {col:25} | Type: {str(dtype):10} | Unique: {unique_vals:4}")
            elif col in ['duration', 'installment_rate', 'present_residence_since', 'num_curr_loans', 'num_people_provide_maint']:
                try:
                    numeric_col = pd.to_numeric(self.cleaned_data[col], errors='coerce')
                    if numeric_col.notna().sum() > 0:
                        min_val = numeric_col.min()
                        max_val = numeric_col.max()
                        print(f"  {col:25} | Type: {str(dtype):10} | Unique: {unique_vals:4} | Range: {min_val:.0f}-{max_val:.0f}")
                    else:
                        print(f"  {col:25} | Type: {str(dtype):10} | Unique: {unique_vals:4}")
                except:
                    print(f"  {col:25} | Type: {str(dtype):10} | Unique: {unique_vals:4}")
            else:
                print(f"  {col:25} | Type: {str(dtype):10} | Unique: {unique_vals:4}")
        
        # loan_amt statistics
        try:
            loan_amounts = pd.to_numeric(self.cleaned_data['loan_amt'], errors='coerce')
            print(f"\nLoan Amount Statistics:")
            print(f"  Count: {loan_amounts.notna().sum()}")
            print(f"  Min: {loan_amounts.min():.2f}")
            print(f"  Max: {loan_amounts.max():.2f}")
            print(f"  Mean: {loan_amounts.mean():.2f}")
            print(f"  Median: {loan_amounts.median():.2f}")
            print(f"  Q1: {loan_amounts.quantile(0.25):.2f}")
            print(f"  Q3: {loan_amounts.quantile(0.75):.2f}")
            print(f"  Std Dev: {loan_amounts.std():.2f}")
        except:
            pass
        
        # Show sample of the data
        print(f"\nFirst 5 rows of cleaned data:")
        print(self.cleaned_data.head().to_string())
        
        # Check for any remaining missing values
        missing_summary = self.cleaned_data.isnull().sum()
        if missing_summary.sum() > 0:
            print(f"\nRemaining missing values:")
            for col, missing in missing_summary.items():
                if missing > 0:
                    print(f"  {col}: {missing}")
        else:
            print(f"\n✓ No missing values in cleaned dataset")

def main():
    """Main execution function"""
    print("GERMAN CREDIT DATASET CLEANSER")
    print("="*50)
    
    # Initialize the cleanser
    cleanser = GermanCreditCleanser('german_credit_dirty.csv')
    
    # Load data
    if not cleanser.load_data():
        return
    
    # Analyze data
    analysis = cleanser.analyze_data()
    
    if cleanser.clean_data():
            # Display report
            cleanser.display_cleaning_report()
            
            # Show data summary
            cleanser.get_data_summary()
            
            # Save cleaned data
            cleanser.save_cleaned_data()
            
            print(f"\n🎉 Data cleaning completed successfully!")
            print(f"   The cleaned dataset with {len(cleanser.cleaned_data)} rows is ready for use.")
    else:
        print("Data cleaning failed.")

if __name__ == "__main__":
    main()