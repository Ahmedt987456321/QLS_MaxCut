import inspect
from src.gain_cache import GainCache
# print all methods
for name, method in inspect.getmembers(GainCache, predicate=inspect.isfunction):
    print(f"\n--- {name} ---")
    try:
        print(inspect.getsource(method))
    except:
        print("(source unavailable)")
