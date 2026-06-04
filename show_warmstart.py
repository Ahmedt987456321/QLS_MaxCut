import inspect
from src.adaptive_qls import adaptive_qls
src = inspect.getsource(adaptive_qls)
# show lines 33-46 (the warm-start pool)
lines = src.split('\n')
for i,line in enumerate(lines[33:50], start=33):
    print(f"  {i:4d}: {line}")
