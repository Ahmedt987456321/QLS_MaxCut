import inspect
from src.adaptive_qls import adaptive_qls
src = inspect.getsource(adaptive_qls)
lines = src.split('\n')
print("Lines 1-35 of adaptive_qls (initialisation):")
for i,line in enumerate(lines[:35], start=0):
    print(f"  {i:4d}: {line}")
