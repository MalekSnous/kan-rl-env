import sys, os, numpy as np
sys.path.insert(0, '.')
os.environ.setdefault('DATA_DIR', 'data')
os.environ.setdefault('SOLUTION_DIR', 'solution')
os.environ.setdefault('TRACE_DIR', '/tmp/kan_traces')
from env_api.kan_env import predict_from_expr
_EXPRESSIONS = {
    "A": '1.948*(0.5098*x1 + 0.3839*x2 - 1)**2 - 8.572*np.sin(0.3729*x1 + 0.1978*(0.6033 - x2)**2 - 3.986) + 3.07*np.sin(0.7506*x1 - 1.2*x2 + 2.264) + 3.359',
    "B": '-1.317*np.sin(-0.7832*x1 + 0.3618*x2 + 5.005) + 0.6068*np.sin(0.257*x1 + 0.5835*x2 + 4.926) + 13.18*np.sin(0.2699*(0.5255 - x2)**2 + 0.244*np.sin(0.4513*x1 + 0.2737) + 7.028) - 9.609',
    "C": '-0.0002773*x1 - 0.0001229*x2 + 7.296*np.exp(0.001704*x1 - 1.337*np.exp(0.3394*x2)) - 0.01786*np.exp(0.004727*x2 + 4.769*np.sqrt(1 - 0.2467*x1)) + 1.836',
    "D": '-1.169e-8*x1 + 8.297e-5*x2 - 25.15*np.sin(0.1733*x1 - 0.1072*x2 + 1.379) + 25.39',
}
def discover_law(dataset_id): return _EXPRESSIONS[dataset_id]
def predict(dataset_id, X): return predict_from_expr(_EXPRESSIONS[dataset_id], X)
