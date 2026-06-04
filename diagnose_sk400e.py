import inspect
from src.metrics import Metrics
from src.adaptive_qls import adaptive_qls

# check record_cut
print("Metrics.record_cut:")
print(inspect.getsource(Metrics.record_cut))

# check adaptive_qls source for record_cut calls
src = inspect.getsource(adaptive_qls)
lines = src.split('\n')
print("\nadaptive_qls lines mentioning record_cut or best_cut:")
for i,line in enumerate(lines):
    if 'record_cut' in line or 'best_cut' in line:
        print(f"  {i:4d}: {line}")
