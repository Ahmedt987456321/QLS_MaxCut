from src.selectors import select_beta_routed, get_selector
from src.graph import load_gset
from src.gain_cache import GainCache
from src.local_search import random_cut, one_flip_ls
import numpy as np

# Test 1: selector is callable and registered
sel = get_selector('beta_routed')
print(f'Test 1 - registered: {sel.__name__}')

# Test 2: routes G11 to Fiedler (beta_e=0.010, asym=0.000)
G11 = load_gset('data/gset/G11.txt')
rng = np.random.default_rng(42)
x = random_cut(G11, rng)
gc = GainCache(); gc.update(G11, x)
S = select_beta_routed(G11, gc, 400, rng=rng, x=x)
route = G11.graph.get('_beta_routing_cache', {}).get('route', 'unknown')
beta_e = G11.graph.get('_beta_routing_cache', {}).get('beta_e', -1)
print(f'Test 2 - G11: route={route}, beta_e={beta_e:.4f}')
print(f'  Expected: route=fiedler, beta_e~0.010')
print(f'  Correct: {route == "fiedler"}')

# Test 3: routes G1 to FConn (beta_e=0.458)
G1 = load_gset('data/gset/G1.txt')
x1 = random_cut(G1, rng)
gc1 = GainCache(); gc1.update(G1, x1)
S1 = select_beta_routed(G1, gc1, 400, rng=rng, x=x1)
route1 = G1.graph.get('_beta_routing_cache', {}).get('route', 'unknown')
beta_e1 = G1.graph.get('_beta_routing_cache', {}).get('beta_e', -1)
print(f'Test 3 - G1: route={route1}, beta_e={beta_e1:.4f}')
print(f'  Expected: route=fconn, beta_e~0.458')
print(f'  Correct: {route1 == "fconn"}')

# Test 4: cache works (second call should not recompute)
import time
t0 = time.time()
S_cached = select_beta_routed(G11, gc, 400, rng=rng, x=x)
t1 = time.time()
print(f'Test 4 - cache: second call took {(t1-t0)*1000:.1f}ms (should be <100ms)')
