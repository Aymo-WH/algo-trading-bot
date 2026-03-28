import timeit

setup = """
import pandas as pd
df = pd.DataFrame(columns=['Close', 'RSI', 'MACD', 'Sentiment_Score', 'BB_Upper', 'BB_Lower', 'ATR', 'Other'])
required_columns = ['Close', 'RSI', 'MACD', 'Sentiment_Score', 'BB_Upper', 'BB_Lower', 'ATR']
"""

loop_time = timeit.timeit("all(col in df.columns for col in required_columns)", setup=setup, number=100000)
set_time = timeit.timeit("set(required_columns).issubset(df.columns)", setup=setup, number=100000)

print(f"Baseline (loop): {loop_time:.4f} seconds")
print(f"Optimized (set): {set_time:.4f} seconds")
print(f"Improvement: {(loop_time - set_time) / loop_time * 100:.2f}%")
