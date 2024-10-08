import numpy as np

EPS = 1e-5
GHOST_FACE_SIDE = (np.iinfo(np.int32).min, np.iinfo(np.int32).min)
GHOST_INDEX = np.iinfo(np.int32).max 
INF = np.finfo(np.float64).max

# constants for QEM
GHOST_HALF_EDGE = np.iinfo(np.int32).max - 5
INVALID_HALF_EDGE = np.iinfo(np.int32).max - 1
INVALID_VERTEX_INDEX = np.iinfo(np.int32).max - 2
INF_COST = np.finfo(np.float64).max - 1.0
HAS_BEEN_COLLAPSED = np.iinfo(np.int32).max - 3
INVALID_EDGE = np.iinfo(np.int32).max - 4
