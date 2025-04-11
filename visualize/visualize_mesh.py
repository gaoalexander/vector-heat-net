import igl
import numpy as np
import polyscope as ps
from scipy.stats import stats
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def visualize_mesh(v, f, normals=None, scalars=None, vectors=None, name="mesh", position=np.array([0, 0, 0]), defined_on="vertices"):
    ps.set_ground_plane_mode("none")
    ps.register_surface_mesh(name, v, f, enabled=True)
    ps.get_surface_mesh(name).set_position(position)
    ps.get_surface_mesh(name).set_edge_width(0.1)

    if scalars is not None:
        ps.get_surface_mesh(name).add_scalar_quantity("scalar_data", scalars, defined_on=defined_on)
    if vectors is not None:
        ps.get_surface_mesh(name).add_vector_quantity("vector_data", vectors, vectortype='standard', defined_on=defined_on)
    if normals is not None:
        ps.get_surface_mesh(name).add_vector_quantity("normals", normals)

# mesh_filepath = "/Users/alexandergao/git/unified-geometry-representation/data/sig17_seg_benchmark/meshes/train/faust/volumetric_normal_test.obj"
#
# v, _, n, f, _, _ = igl.read_obj(mesh_filepath)
# print(n)
# ps.init()
#
# ps.register_point_cloud("cloud", v)
# ps.get_point_cloud("cloud").add_vector_quantity("normals", n)
# ps.show()
