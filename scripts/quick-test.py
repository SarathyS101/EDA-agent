#!/usr/bin/env python3
"""
Quick test to verify core Python functionality
"""

print("🧪 Quick Test - Agentic EDA Core")
print("=" * 40)

# Test 1: Basic imports
print("Testing imports...")
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✅ Core data science libraries imported")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test 2: Create sample data
print("Creating sample data...")
np.random.seed(42)
data = {
    'age': np.random.randint(18, 80, 100),
    'salary': np.random.normal(50000, 15000, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
}
df = pd.DataFrame(data)
print(f"✅ Created DataFrame with {df.shape[0]} rows, {df.shape[1]} columns")

# Test 3: Basic EDA
print("Running basic EDA...")
print(f"   - Data types: {len(df.dtypes)} columns")
print(f"   - Missing values: {df.isnull().sum().sum()}")
print(f"   - Numeric columns: {len(df.select_dtypes(include=['number']).columns)}")
print(f"   - Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")

# Test 4: Simple visualization
print("Testing visualization...")
try:
    plt.figure(figsize=(6, 4))
    df['age'].hist()
    plt.title('Age Distribution')
    plt.savefig('test_chart.png')
    plt.close()
    print("✅ Visualization created: test_chart.png")
    
    # Clean up
    import os
    if os.path.exists('test_chart.png'):
        os.remove('test_chart.png')
        print("✅ Cleaned up test file")
        
except Exception as e:
    print(f"⚠️  Visualization test failed: {e}")

# Test 5: OpenAI (if available)
try:
    import openai
    import os
    
    if os.environ.get('OPENAI_API_KEY'):
        print("✅ OpenAI library available with API key")
    else:
        print("⚠️  OpenAI library available but no API key")
        
except ImportError:
    print("❌ OpenAI library not available")

print("\n🎉 Core functionality test completed!")
print("\nNext steps:")
print("1. Run 'cd scripts && python3 test-python.py' for full test")
print("2. Run 'npm run dev' to start the web application")
print("3. Visit http://localhost:3000")