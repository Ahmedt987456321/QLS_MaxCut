import inspect
from src.adaptive_qls import adaptive_qls
src = inspect.getsource(adaptive_qls)
lines = src.split('\n')
print("Lines 35-75 of adaptive_qls:")
for i,line in enumerate(lines[35:75], start=35):
    print(f"  {i:4d}: {line}")
