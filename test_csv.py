import pandas as pd
import json

def test_csv_loading():
    """Test script to validate CSV loading"""
    try:
        # Try to load the CSV with error handling
        df = pd.read_csv('questions.csv', encoding='utf-8', on_bad_lines='skip')
        print(f"✅ Successfully loaded {len(df)} rows from CSV")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Check for the expected number of columns
        expected_cols = 13
        actual_cols = len(df.columns)
        if actual_cols == expected_cols:
            print(f"✅ Column count matches expected ({expected_cols})")
        else:
            print(f"⚠️ Column count mismatch: expected {expected_cols}, got {actual_cols}")
        
        # Test parsing a few answer JSON fields
        print("\n🧪 Testing JSON parsing...")
        for i in range(min(3, len(df))):
            try:
                answer_data = json.loads(df.iloc[i]['answer'])
                print(f"✅ Row {i+1}: JSON parsing successful")
                print(f"   - Question: {answer_data.get('question', 'N/A')[:50]}...")
                print(f"   - Answer options: {len(answer_data.get('answers', []))}")
            except json.JSONDecodeError as e:
                print(f"❌ Row {i+1}: JSON parsing failed - {e}")
        
        # Check difficulty values
        print(f"\n📈 Difficulty range: {df['overall_difficulty'].min():.1f} - {df['overall_difficulty'].max():.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing CSV loading and parsing...")
    test_csv_loading()