import argparse
import json
import os
from pathlib import Path

import h5py
import igl
import numpy as np
import pymeshlab
from tqdm import tqdm


def project_points_onto_surface(source_vertices, source_quad_faces, barycenters):
    """

    Parameters
    ----------
    source_vertices: np.array
        dim=(n_verts_source, 3)
    source_quad_faces: np.array
        dim=(n_faces_source, 3)
    barycenters
        dim=(n_faces_processed, 3)
    Returns
    -------
    indices: np.array
        dim=(n_faces_processed)
        for each processed triangle face barycenter, the index of nearest face on the source quad mesh.
    nearest_points: np.array
        dim=(n_faces_processed, 3)
        for each processed triangle face barycenter, the coordinates of nearest point on the source quad mesh.
    """
    source_tri_faces = igl.polygon_mesh_to_triangle_mesh(source_quad_faces)
    distances, indices, nearest_points = igl.point_mesh_squared_distance(
        barycenters, source_vertices, source_tri_faces
    )
    return (
        indices // 2,
        nearest_points,
    )  # Divide indices by 2 to convert triangle idx into corresponding quad index


def point_to_edge_distance(v1, v2, point):
    """
    Parameters
    ----------
    v1: np.array
        dim=(3)
        quad vertex (adjacent to v2)
    v2: np.array
        dim=(3)
        quad vertex (adjacent to v1)
    point: np.array
        dim=(3)
        point within interior of quad face
    Returns
    -------
    distance from interior point to edge
    """
    return np.linalg.norm(np.cross(point - v1, v2 - v1)) / np.linalg.norm(v2 - v1)


def distances_to_quad_edges(indices, projected, source_vertices, source_quad_faces):
    """
    For each proejcted point (tri barycenter projected onto quad mesh faces),
    compute the distance to each edge of the quad mesh.

    Parameters
    ----------
    indices: np.array
        dim=(n_faces_processed)
        for each processed triangle face barycenter, the index of nearest face on the source quad mesh.
    projected: np.array
        dim=(n_faces_processed, 3)
        for each processed triangle face barycenter, the coordinates of nearest point on the source quad mesh.
    source_vertices: np.array
        dim=(n_verts_source, 3)
    source_quad_faces: np.array
        dim=(n_faces_source, 3)

    Returns
    -------
    edges: np.array
        dim=(n_faces_processed, 4, 3)
        for each processed triangle face barycenter, the 4 edges (expressed as vectors) of the
        corresponding nearest face on the source quad mesh.
    distances: np.array
        dim=(n_faces_processed, 4, 1)
        for each processed triangle face barycenter, the distance (as scalar value) to each of
        the 4 edges of the corresponding nearest face on the source quad mesh.
    """
    edges = []  # Dim: len(Projected) x 4 x 3
    distances = []  # Dim: len(Projected) x 4

    for i, idx in enumerate(indices):
        """
        Example quad:
        v3--------->v2
        ^           ^
        |   .Point  |
        |           |
        v0--------->v1
        """
        source_quad_face = source_quad_faces[idx]
        v0 = source_vertices[source_quad_face[0]]
        v1 = source_vertices[source_quad_face[1]]
        v2 = source_vertices[source_quad_face[2]]
        v3 = source_vertices[source_quad_face[3]]

        distances.append(
            [
                point_to_edge_distance(v0, v1, projected[i]),
                point_to_edge_distance(v1, v2, projected[i]),
                point_to_edge_distance(v2, v3, projected[i]),
                point_to_edge_distance(v3, v0, projected[i]),
            ]
        )

        edges.append([v1 - v0, v2 - v1, v2 - v3, v3 - v0])

    return np.array(edges), np.array(distances)


def compute_uv(indices, projected, source_vertices, source_quad_faces):
    """
    Given projected points, compute the ground truth vectors.
    Parameters
    ----------
    indices: np.array
        dim=(n_faces_processed)
        for each processed triangle face barycenter, the index of nearest face on the source quad mesh.
    projected: np.array
        dim=(n_faces_processed, 3)
        for each processed triangle face barycenter, the coordinates of nearest point on the source quad mesh.
    source_vertices: np.array
        dim=(n_verts_source, 3)
    source_quad_faces: np.array
        dim=(n_faces_source, 3)

    Returns
    -------
    u, v (global coordinate frame)
    """
    edges, distances = distances_to_quad_edges(
        indices, projected, source_vertices, source_quad_faces
    )
    u = (
            np.expand_dims(distances[:, 2] / (distances[:, 0] + distances[:, 2]), axis=1)
            * edges[:, 0, :]
            + np.expand_dims(distances[:, 0] / (distances[:, 0] + distances[:, 2]), axis=1)
            * edges[:, 2, :]
    )
    v = (
            np.expand_dims(distances[:, 3] / (distances[:, 1] + distances[:, 3]), axis=1)
            * edges[:, 1, :]
            + np.expand_dims(distances[:, 1] / (distances[:, 1] + distances[:, 3]), axis=1)
            * edges[:, 3, :]
    )
    return u, v


def compute_ground_truth_uv(
        output_vertices, output_tri_faces, source_vertices, source_quad_faces
):
    """
    Project processed tri mesh barycenters onto nearest quad face to compute the "ground truth" directional vectors.

    Parameters
    ----------
    output_vertices: np.array
        dim=(n_verts_processed, 3)
    output_tri_faces: np.array
        dim=(n_faces_processed, 3)
    source_vertices: np.array
        dim=(n_verts_source, 3)
    source_quad_faces: np.array
        dim=(n_faces_source, 3)

    Returns
    -------
    dict containing barycenters, u, v (global coordinate frame)
    """
    barycenters = igl.barycenter(output_vertices, output_tri_faces)
    indices, projected = project_points_onto_surface(
        source_vertices, source_quad_faces, output_vertices
    )
    u, v = compute_uv(indices, projected, source_vertices, source_quad_faces)
    return {"barycenters": barycenters.tolist(), "u": u.tolist(), "v": v.tolist()}


def save_to_h5(savepath, datadict):
    with h5py.File(savepath, "w") as file:
        # Create a group similar to a dictionary key
        group = file.create_group("data")
        for key in list(datadict.keys()):
            group.create_dataset(
                key, data=datadict[key], compression="gzip", compression_opts=9
            )


def save_to_json(savepath, datadict):
    with open(savepath.replace("h5", "json"), "w") as file:
        json.dump(datadict, file)


def save_processed_triangle_mesh(
        savepath, vertices, faces, source_directory, output_directory
):
    savepath = savepath.replace(source_directory, output_directory)
    igl.write_obj(savepath, vertices, faces)


def normalize(verts):
    x_min = verts[:, 0].min()
    x_max = verts[:, 0].max()
    y_min = verts[:, 1].min()
    y_max = verts[:, 1].max()
    z_min = verts[:, 2].min()
    z_max = verts[:, 2].max()

    verts[:, 0] -= ((x_max + x_min) / 2)
    verts[:, 1] -= ((y_max + y_min) / 2)
    verts[:, 2] -= ((z_max + z_min) / 2)

    largest_axis_scale = max((x_max - x_min), max((y_max - y_min), (z_max - z_min)))
    verts /= largest_axis_scale
    return verts


def preprocess_training_data(source_directory, output_directory, scale_percentage):
    """
    This script preprocesses a batch of GT quad mesh data to train retopo model.
    The main workflow accepts an arbitrary triangulated mesh, and cmoputes its ground truth frame field.
    It does so by computing the barycenter of each triangle, and projecting that point onto an ideal quad mesh of the
    (roughly) same shape.  Then, it computes the per-face frame by interpolating between the horizontal and vertical
    edges of that corresponding quad face.

    Assume that input is:
    1.) a quad-only mesh produced by applying Catmull-Clark subdivision (2-3 iterations) to original mesh.
    2.) a triangle mesh produced by remeshing original good mesh, triangulating, then decimating to target # of faces.

    Parameters
    ----------
    source_directory: path
    output_directory: path
    target_num_faces: int
        target number of triangular faces
    """
    os.makedirs(output_directory, exist_ok=True)

    filepaths = sorted(list(Path(source_directory).glob("*.obj")))
    filepaths = [str(filepath) for filepath in filepaths]

    for i, filepath in tqdm(enumerate(filepaths)):
        print(f"Processing {filepath}...")
        head, tail = os.path.split(filepath)
        mesh_output_path = os.path.join(output_directory, tail)
        target_field_output_path = os.path.join(
            output_directory, tail.replace("obj", "h5")
        )

        if not os.path.exists(mesh_output_path):
            source_vertices, _, _, source_quad_faces, _, _ = igl.read_obj(filepath)
            source_vertices = normalize(source_vertices)

            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(source_vertices, source_quad_faces))
            ms.meshing_poly_to_tri()
            ms.meshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.Percentage(scale_percentage),
            )
            # ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_num_faces,
            #                                             qualitythr=0.99,
            #                                             preservenormal=True)
            output_vertices, output_tri_faces = (
                ms.current_mesh().vertex_matrix(),
                ms.current_mesh().face_matrix(),
            )

            processed = compute_ground_truth_uv(
                output_vertices, output_tri_faces, source_vertices, source_quad_faces
            )

            output_data = {"barycenters": [], "u": [], "v": []}
            for key in list(processed.keys()):
                output_data[key].append(processed[key])

            save_to_h5(target_field_output_path, output_data)
            save_to_json(target_field_output_path, output_data)
            save_processed_triangle_mesh(
                mesh_output_path,
                output_vertices,
                output_tri_faces,
                source_directory,
                output_directory,
            )
        else:
            print("Skipping file: {}".format(mesh_output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_quad_mesh_dir',
                        type=str,
                        help='path to directory containing source quadrilateral meshes',
                        default='experiments/quad_meshing/data/example_custom_dataset/quad')
    parser.add_argument('--output_dir',
                        type=str,
                        help='path to directory where preprocessed meshes + computed GT vector fields will be written',
                        default='experiments/quad_meshing/data/example_custom_dataset/preprocessed/train')
    parser.add_argument('--scale_percentage',
                        type=float,
                        help='edge length scale percentage for triangle remeshing (via pymeshlab)',
                        default=1.0)
    args = parser.parse_args()
    preprocess_training_data(args.source_quad_mesh_dir, args.output_dir, args.scale_percentage)
