import inspect
from src.gain_cache import GainCache
print(inspect.signature(GainCache.__init__))
print(inspect.getsource(GainCache.__init__))
