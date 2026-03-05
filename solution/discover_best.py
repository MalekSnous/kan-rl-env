import sys, os, numpy as np
sys.path.insert(0, '.')
os.environ.setdefault('DATA_DIR', 'data')
os.environ.setdefault('SOLUTION_DIR', 'solution')
os.environ.setdefault('TRACE_DIR', '/tmp/kan_traces')
from env_api.kan_env import predict_from_expr
_EXPRESSIONS = {
    "A": '1.945*(0.522*x1 + 0.3918*x2 - 1)**2 + 7.748*np.sin(0.4894*x1 + 0.1261*(0.104 - x2)**2 - 7.69) - 6.9*np.sin(6.556*np.sqrt(1 - 0.1674*x2) + 0.1049*np.exp(0.7969*x1) - 1.156) + 1.454',
    "B": '0.1942*(0.7493*x2 - 0.5912*(1 - 0.4632*x1)**2 + 1)**2 - 5.177*(0.02777*np.exp(1.477*x2) - 0.2624*np.sin(0.5604*x1 + 3.097) - 1)**2 + 1.817*np.sin(-1.591*np.sin(0.5273*x1 + 1.864) + 0.475*np.sin(0.879*x2 - 5.098) + 8.748) + 3.061',
    "C": '0.03324*x1 - 0.2748*x2 + 4.924*np.sqrt(-0.003981*x2 - np.sin(0.2415*x1 + 8.215) + 0.9906) - 0.5592*np.sin(0.9187*x1 + 3.856) - 1.359*np.sin(0.261*x2 - 1.292) - 1.638',
    "D": '3.114*(-0.3545*x1 + 0.0251*np.sin(1.633*x2 - 3.961) + 1)**2 - 0.002547*np.exp(1.578*np.sin(0.9723*x1 + 1.761) - 5.222*np.sin(0.4395*x2 + 4.623)) - 0.1453*np.sin(0.0432*np.exp(1.287*x2) + 5.763*np.sin(0.7612*x1 - 8.492) + 7.311) + 0.02942',
}
def discover_law(dataset_id): return _EXPRESSIONS[dataset_id]
def predict(dataset_id, X): return predict_from_expr(_EXPRESSIONS[dataset_id], X)
