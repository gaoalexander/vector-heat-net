import numpy as np
from .global_variables import GHOST_HALF_EDGE
from .twin import twin
from .next import next
from .half_edge_lengths import half_edge_lengths
from .vertex_normals import vertex_normals
from .face_side_to_half_edge_index import face_side_to_half_edge_index
from .tail_vertex import tail_vertex
from .tip_vertex import tip_vertex
from .opposite_corner_angle import opposite_corner_angle
from .gram_schmidt_orthogonalization import gram_schmidt_orthogonalization

def half_edge_tangent_vectors(V,F,twins,tips,return_basis_vectors=False, alternate_basis=False):
    """
    for each half edge in the mesh, this function computes the tangent vector (represented as a 2D array of complex number) of that half edge in the tangent plane of its tail vertex

    Parameters
    ==========
    V: 2-d array
        (V,3) vertex locations
    F: 2-d array
        (F,3) face list
    twins: 1-d array
        (F*3,) array of twin half edge indices
    tips: 1-d array
        (F*3,) array of tip vertex indices of each half edgge
    return_basis_vectors: bool
        True/False of whether to return the tangent plane basis vectors in 3D (for visualization) 

    Returns
    =======
    tangent_vectors: 1-d array of complex numbers
        (F*3,) array of tangent representation of the half edge (using complex numbers)
    v2he_x: 1-d array
        (V,) map from each vertex to the half-edge index representing basis x
    basis_x, basis_y: 2-d array
        (V,3) arrays representing tangent plane basis vectors in 3D (for visualization)
    """
    nV = V.shape[0]
    nF = F.shape[0]
    hEl = half_edge_lengths(V,F)
    VN = vertex_normals(V,F)

    V_visited = np.full((nV,), False, dtype=bool)
    v2he_x = np.zeros(nV, dtype=int)
    # tangent_vectors = np.empty((nF*3,2), dtype=np.float64) # 2-d tangent vector for each half edge
    hE_tangent_vectors = np.empty((nF*3,), dtype=np.csingle) # tangent vector for each half edge, represented as compelx numbers
    basis_x = np.empty((nV,3), dtype=np.float64) # tangent basis vector x
    basis_y = np.empty((nV,3), dtype=np.float64) # tangetn basis vector y
    basis_n = np.empty((nV,3), dtype=np.float64) # tangetn basis vector y
    for f in range(nF):
        for s in range(3):
            he = face_side_to_half_edge_index((f, s))
            v = tail_vertex(tips, he)
            if V_visited[v] == False: # not visited yet
                V_visited[v] = True # set it to visited

                # get list of one-ring face sides
                he_list, is_boundary_vertex = half_edge_one_ring_possibly_boundary(twins,he, alternate_basis=alternate_basis)

                # comput vertex angles
                angles_v = np.zeros(len(he_list))
                for ii in range(len(angles_v)):
                    angles_v[ii] = opposite_corner_angle(hEl, next(he_list[ii]))

                # normalize the angular coordinates 
                # - scale angles_v for interior vertices so that it sums up to 2pi
                # - scale (angles_v+pi) for boundary vertices so that it sums up to 2pi 
                if is_boundary_vertex: # if boundary vertex
                    angles_v = angles_v / (angles_v.sum()+np.pi) * np.pi*2
                else: # if interior vertex
                    angles_v = angles_v / (angles_v.sum()) * np.pi*2

                # put these angles back to tangent vectors
                ang_coord = np.cumsum(angles_v) 
                ang_coord = ang_coord[:-1] # delete the last one
                ang_coord = np.insert(ang_coord, 0, 0.0) # inser 0 to the first one

                # assign tangent vectors
                for ii in range(len(he_list)):
                    he_ii = he_list[ii]
                    length_ii = hEl[he_ii]
                    angle_ii = ang_coord[ii] 
                    # tangent_vector = length_ii * np.array([np.cos(angle_ii) + np.sin(angle_ii)])
                    tangent_vector = length_ii * (np.cos(angle_ii) + 1j * np.sin(angle_ii))
                    hE_tangent_vectors[he_list[ii]] = tangent_vector

                # store the basis half edge vector (x) for vertex vi
                v2he_x[v] = he_list[0]

                if return_basis_vectors:
                    # compute tangent bases
                    n = VN[v,:]
                    v_tip = tip_vertex(tips, he_list[0])
                    x = V[v_tip,:] - V[v,:] 
                    x = x / np.linalg.norm(x)
                    x = gram_schmidt_orthogonalization(x, n)
                    y = np.cross(n, x)
                    
                    basis_x[v,:] = x
                    basis_y[v,:] = y
                    basis_n[v,:] = n
    if return_basis_vectors:
        basis_x = basis_x / np.linalg.norm(basis_x, axis=1)[:, None]
        basis_y = basis_y / np.linalg.norm(basis_y, axis=1)[:, None]
        basis_n = basis_n / np.linalg.norm(basis_n, axis=1)[:, None]
        return hE_tangent_vectors, v2he_x, basis_x, basis_y, basis_n
    else:
        return hE_tangent_vectors, v2he_x
    
def half_edge_one_ring_possibly_boundary(twins,he_start, alternate_basis=False):
    """
    Get one-ring face sides of a vertex (similar to the face_side_one_ring.py with a different set of inputs).
    """
    he = he_start
    one_ring = [he]
    is_boundary_vertex = False
    while True:
        he = twin(twins, next(next(he)))
        if he == he_start:
            # this is an interior vertex, so it reaches the starting fs
            break
        if he == GHOST_HALF_EDGE: # hit boundary, then go clock wise
            is_boundary_vertex = True
            he = he_start
            while True:
                if twin(twins, he) == GHOST_HALF_EDGE:
                    break
                he = next(twin(twins, he))
                one_ring.insert(0,he)
            break
        one_ring.append(he)
    if alternate_basis:
        one_ring = np.roll(np.array(one_ring), 1).tolist()

    return one_ring, is_boundary_vertex
