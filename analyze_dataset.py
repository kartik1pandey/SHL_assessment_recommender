"""
Analyze the company-provided GenAI dataset
"""

import pandas as pd
import json

def analyze_genai_dataset():
    """Analyze the Gen_AI Dataset.xlsx file."""
    
    print("=" * 60)
    print("ANALYZING COMPANY-PROVIDED GENAI DATASET")
    print("=" * 60)
    
    # Read all sheets
    df_dict = pd.read_excel('Gen_AI Dataset.xlsx', sheet_name=None)
    
    print(f"📊 Found {len(df_dict)} sheets: {list(df_dict.keys())}")
    
    # Analyze Train Set
    train_df = df_dict['Train-Set']
    print(f"\n🚂 TRAIN SET:")
    print(f"   Shape: {train_df.shape}")
    print(f"   Columns: {list(train_df.columns)}")
    
    print(f"\n📝 Sample Training Queries:")
    for i in range(min(5, len(train_df))):
        query = train_df.iloc[i]['Query']
        url = train_df.iloc[i]['Assessment_url']
        print(f"   {i+1}. Query: {query[:80]}...")
        print(f"      URL: {url}")
        print()
    
    # Analyze Test Set
    test_df = df_dict['Test-Set']
    print(f"\n🧪 TEST SET:")
    print(f"   Shape: {test_df.shape}")
    print(f"   Columns: {list(test_df.columns)}")
    
    print(f"\n📝 Sample Test Queries:")
    for i in range(min(5, len(test_df))):
        query = test_df.iloc[i]['Query']
        if 'Assessment_url' in test_df.columns:
            url = test_df.iloc[i]['Assessment_url']
            print(f"   {i+1}. Query: {query[:80]}...")
            print(f"      URL: {url}")
        else:
            print(f"   {i+1}. Query: {query[:80]}...")
        print()
    
    # Check for unique queries
    print(f"\n📈 DATASET STATISTICS:")
    print(f"   Unique training queries: {train_df['Query'].nunique()}")
    print(f"   Unique test queries: {test_df['Query'].nunique()}")
    
    if 'Assessment_url' in train_df.columns:
        print(f"   Unique training URLs: {train_df['Assessment_url'].nunique()}")
    if 'Assessment_url' in test_df.columns:
        print(f"   Unique test URLs: {test_df['Assessment_url'].nunique()}")
    
    return train_df, test_df

def extract_assessment_names_from_urls(df):
    """Extract assessment names from URLs."""
    assessment_names = []
    
    for url in df['Assessment_url'].unique():
        # Try to extract assessment name from URL
        if 'product' in url:
            # Extract product name from URL
            parts = url.split('/')
            for i, part in enumerate(parts):
                if part == 'products' and i + 1 < len(parts):
                    assessment_names.append(parts[i + 1])
                    break
        else:
            assessment_names.append(url.split('/')[-1])
    
    return assessment_names

if __name__ == "__main__":
    train_df, test_df = analyze_genai_dataset()
    
    # Save processed datasets
    print(f"\n💾 SAVING PROCESSED DATASETS:")
    
    # Save as CSV for easier processing
    train_df.to_csv('data/processed/train_set.csv', index=False)
    test_df.to_csv('data/processed/test_set.csv', index=False)
    
    print(f"   ✅ Saved train_set.csv ({len(train_df)} rows)")
    print(f"   ✅ Saved test_set.csv ({len(test_df)} rows)")
    
    # Extract unique assessments
    if 'Assessment_url' in train_df.columns:
        unique_assessments = train_df['Assessment_url'].unique()
        print(f"\n🎯 UNIQUE ASSESSMENTS IN TRAINING DATA:")
        for i, url in enumerate(unique_assessments[:10]):  # Show first 10
            print(f"   {i+1}. {url}")
        
        if len(unique_assessments) > 10:
            print(f"   ... and {len(unique_assessments) - 10} more")
    
    print(f"\n✅ Dataset analysis complete!")