from src.experiment import run_experiment
from src.graph import load_gset
from src.adaptive_qls import adaptive_qls
from src.selectors import select_frustrated_connected
from src.backends import backend_neal, backend_sqa
from pathlib import Path

G = load_gset("data/gset/G11.txt")
bk = 564

methods = [
    {'name': 'AQLS-neal', 'fn': adaptive_qls,
     'kwargs': {'G': G, 'budget_seconds': 30,
                'selector': select_frustrated_connected,
                'backend': backend_neal,
                'k_min': 400, 'k_max': 400,
                'n_reads': 200, 'best_known': bk,
                'acceptance': 'lookahead'}},
    {'name': 'AQLS-sqa', 'fn': adaptive_qls,
     'kwargs': {'G': G, 'budget_seconds': 30,
                'selector': select_frustrated_connected,
                'backend': backend_sqa,
                'k_min': 400, 'k_max': 400,
                'n_reads': 200, 'best_known': bk,
                'acceptance': 'lookahead'}},
]

run_experiment(G=G, methods=methods, n_trials=5, best_known=bk,
               results_dir=Path('results'), experiment_name='try_sqa_vs_neal')