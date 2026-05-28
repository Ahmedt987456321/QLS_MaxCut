"""
GainCache — explicit gain array management with invalidation.
Prevents stale gain bugs by hard-crashing if gains are read after
a solution change without recomputation.
"""


class GainCache:
    """
    Maintains flip-gain values for all vertices.
    gain[v] = improvement in cut value if vertex v is flipped.
    Positive gain means flipping v improves the cut.
    """

    def __init__(self):
        self.gain = {}
        self.valid = False

    def update(self, G, x):
        """Full recompute — O(|E|). Call when solution changes significantly."""
        self.gain = {}
        for v in G.nodes():
            self.gain[v] = self._compute_gain(G, x, v)
        self.valid = True

    def incremental_update(self, G, x, flipped_v):
        """Partial recompute after one flip — O(deg(v))."""
        assert self.valid, "Cannot incrementally update invalid cache"
        self.gain[flipped_v] = self._compute_gain(G, x, flipped_v)
        for u in G.neighbors(flipped_v):
            self.gain[u] = self._compute_gain(G, x, u)

    def invalidate(self):
        """Call after any accepted QLS proposal — forces full recompute next use."""
        self.valid = False

    def assert_valid(self):
        """Call at the start of every selector — hard crash if stale."""
        assert self.valid, (
            "GainCache is stale. Call update() before reading gains."
        )

    def best_flip(self):
        """Returns (vertex, gain) for the best improving flip."""
        self.assert_valid()
        best_v = max(self.gain, key=lambda v: self.gain[v])
        return best_v, self.gain[best_v]

    def _compute_gain(self, G, x, v):
        """
        Flip gain for vertex v given current assignment x.
        gain > 0 means flipping v increases the cut.
        """
        gain = 0
        for u in G.neighbors(v):
            w = G[v][u].get('weight', 1.0)
            if x[v] == x[u]:
                gain += w   # currently NOT cut — flipping v would cut this edge
            else:
                gain -= w   # currently cut — flipping v would uncut this edge
        return gain