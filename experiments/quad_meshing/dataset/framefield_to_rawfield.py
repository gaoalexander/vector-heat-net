import json
from pathlib import Path

import click
import numpy as np
import potpourri3d as pp3d
from tqdm import tqdm


@click.group()
def cli():
    """Command-line tools for retopo experiments"""
    pass


def compute_local_axes_triplanar(verts, faces):
    """
    Compute per-face local coordinate frame axes, in 3 unique planar alignments.

    Parameters
    ----------
    verts: np.array
        dim=(n_verts, 3)
    faces: np.array
        dim=(n_faces, 3)

    Returns
    -------
    axis_x_list: list(np.array)
        dim=(3, n_faces, 3)
        First dimension corresponds to the 3 possible choices of planar-alignment.
    axis_y_list: list(np.array)
        dim=(3, n_faces, 3)
        First dimension corresponds to the 3 possible choices of planar-alignment.
    axis_n: np.array
        dim=(n_faces, 3)
        Normal axis reamins the same, independent of choice of planar alignment.
    """
    tri_edge_1 = verts[faces[:, 1], :] - verts[faces[:, 0], :]
    tri_edge_2 = verts[faces[:, 2], :] - verts[faces[:, 0], :]

    axis_n = np.cross(tri_edge_1, tri_edge_2)

    axis_x_list = []
    axis_y_list = []
    zeroed_component = []

    # YZ
    zeroed_component.append(
        np.concatenate(
            (
                np.expand_dims(np.zeros_like(axis_n[:, 0]), axis=1),
                np.expand_dims(axis_n[:, 1], axis=1),
                np.expand_dims(axis_n[:, 2], axis=1),
            ),
            axis=1,
        )
    )
    # XZ
    zeroed_component.append(
        np.concatenate(
            (
                np.expand_dims(axis_n[:, 0], axis=1),
                np.expand_dims(np.zeros_like(axis_n[:, 0]), axis=1),
                np.expand_dims(axis_n[:, 2], axis=1),
            ),
            axis=1,
        )
    )
    # XY
    zeroed_component.append(
        np.concatenate(
            (
                np.expand_dims(axis_n[:, 0], axis=1),
                np.expand_dims(axis_n[:, 1], axis=1),
                np.expand_dims(np.zeros_like(axis_n[:, 0]), axis=1),
            ),
            axis=1,
        )
    )

    axis_x_list.append(np.cross(np.array([[1, 0, 0]]), zeroed_component[0]))
    axis_x_list.append(np.cross(np.array([[0, 1, 0]]), zeroed_component[1]))
    axis_x_list.append(np.cross(np.array([[0, 0, 1]]), zeroed_component[2]))

    axis_y_list.append(np.cross(axis_n, axis_x_list[0]))
    axis_y_list.append(np.cross(axis_n, axis_x_list[1]))
    axis_y_list.append(np.cross(axis_n, axis_x_list[2]))

    # normalize
    axis_n = axis_n / np.expand_dims(np.linalg.norm(axis_n, axis=1), axis=1)
    axis_n = np.expand_dims(axis_n, axis=2)

    # normalize axes
    for i in range(len(axis_x_list)):
        axis_x_list[i] = axis_x_list[i] / np.expand_dims(
            np.linalg.norm(axis_x_list[i], axis=1), axis=1
        )
        axis_y_list[i] = axis_y_list[i] / np.expand_dims(
            np.linalg.norm(axis_y_list[i], axis=1), axis=1
        )
    axis_x_list = np.array(axis_x_list)
    axis_y_list = np.array(axis_y_list)

    return axis_x_list, axis_y_list, axis_n


def make_vectors_consistent(best_directions, secondary_directions):
    similarity1 = np.abs(
        np.dot(best_directions[0:3], secondary_directions[0:3])
    ) + np.abs(np.dot(best_directions[3:6], secondary_directions[3:6]))
    similarity2 = np.abs(
        np.dot(best_directions[0:3], secondary_directions[3:6])
    ) + np.abs(np.dot(best_directions[3:6], secondary_directions[0:3]))

    if similarity2 > similarity1:
        secondary_directions[0:3], secondary_directions[3:6] = (
            secondary_directions[3:6].copy(),
            secondary_directions[0:3].copy(),
        )
    if np.dot(best_directions[0:3], secondary_directions[0:3]) < 0:
        secondary_directions[0:3] = -secondary_directions[0:3].copy()
    if np.dot(best_directions[3:6], secondary_directions[3:6]) < 0:
        secondary_directions[3:6] = -secondary_directions[3:6].copy()
    return secondary_directions


def make_triplanes_consistent(global_directions, axis_n):
    """
    In order to correctly blend the triplanar-aligned predictions, it's required to first align the vector predictions
    made in each of the 3 local coordinate frames, such that rotations of pi radians are accounted for, and (u, v) order
    does not matter.

    Parameters
    ----------
    global_directions: np.array
        dim=(n_faces, 18)
    axis_n: np.array
        dim=(n_faces, 3)

    Returns
    -------
    global_directions: np.array
        dim=(n_faces, 18)
        Similar to input, except that vectors are now aligned as much as possible between the triplanar predictions.
    """
    # (Per face) Compute best plane to define local coordinate frame
    best_plane_idx = np.argmin(np.abs(axis_n), axis=1)
    best_plane_idx = np.squeeze(best_plane_idx)

    best_directions = np.concatenate(
        (
            np.expand_dims(
                global_directions[
                    np.arange(len(global_directions)), best_plane_idx * 6 + 0
                ],
                axis=1,
            ),
            np.expand_dims(
                global_directions[
                    np.arange(len(global_directions)), best_plane_idx * 6 + 1
                ],
                axis=1,
            ),
            np.expand_dims(
                global_directions[
                    np.arange(len(global_directions)), best_plane_idx * 6 + 2
                ],
                axis=1,
            ),
            np.expand_dims(
                global_directions[
                    np.arange(len(global_directions)), best_plane_idx * 6 + 3
                ],
                axis=1,
            ),
            np.expand_dims(
                global_directions[
                    np.arange(len(global_directions)), best_plane_idx * 6 + 4
                ],
                axis=1,
            ),
            np.expand_dims(
                global_directions[
                    np.arange(len(global_directions)), best_plane_idx * 6 + 5
                ],
                axis=1,
            ),
        ),
        axis=1,
    )
    for i in range(len(global_directions)):
        for j in range(3):
            if j != best_plane_idx[i]:
                global_directions[i, j * 6 : j * 6 + 6] = make_vectors_consistent(
                    best_directions[i, :], global_directions[i, j * 6 : j * 6 + 6]
                )
    return global_directions


def get_triplanar_weights(normals):
    weight_yz = 1 - np.abs(normals[:, 0])
    weight_xz = 1 - np.abs(normals[:, 1])
    weight_xy = 1 - np.abs(normals[:, 2])
    weight_sum = weight_xy + weight_xz + weight_yz

    weight_yz /= weight_sum
    weight_xz /= weight_sum
    weight_xy /= weight_sum

    return weight_yz, weight_xz, weight_xy


def get_frame_fields_global_coord(pred_directions, axis_x_list, axis_y_list):
    """
    pred_directions:
         yz              xz              xy
    ux uy vx vy  |  ux uy vx vy  |  ux uy vx vy
    01 02 03 04  |  05 06 07 08  |  09 10 11 12
    """
    # pred_directions = np.array(pred_directions)
    print(pred_directions.shape)
    global_yz_u = (
        np.expand_dims(pred_directions[:, 0], axis=1) * axis_x_list[0]
        + np.expand_dims(pred_directions[:, 1], axis=1) * axis_y_list[0]
    )
    global_yz_v = (
        np.expand_dims(pred_directions[:, 2], axis=1) * axis_x_list[0]
        + np.expand_dims(pred_directions[:, 3], axis=1) * axis_y_list[0]
    )

    global_xz_u = (
        np.expand_dims(pred_directions[:, 4], axis=1) * axis_x_list[1]
        + np.expand_dims(pred_directions[:, 5], axis=1) * axis_y_list[1]
    )
    global_xz_v = (
        np.expand_dims(pred_directions[:, 6], axis=1) * axis_x_list[1]
        + np.expand_dims(pred_directions[:, 7], axis=1) * axis_y_list[1]
    )

    global_xy_u = (
        np.expand_dims(pred_directions[:, 8], axis=1) * axis_x_list[2]
        + np.expand_dims(pred_directions[:, 9], axis=1) * axis_y_list[2]
    )
    global_xy_v = (
        np.expand_dims(pred_directions[:, 10], axis=1) * axis_x_list[2]
        + np.expand_dims(pred_directions[:, 11], axis=1) * axis_y_list[2]
    )

    frame_fields = np.concatenate(
        (global_yz_u, global_yz_v, global_xz_u, global_xz_v, global_xy_u, global_xy_v),
        axis=1,
    )
    return frame_fields


def write_dmat_file(dmat_outpath, u, v):
    with open(dmat_outpath, "a") as f:
        num_faces = u.shape[0]
        lines = ["{} {}\n".format(7, num_faces)]
        mat_body = ""
        print("DEBUG: ")
        print(u.shape, v.shape, num_faces)
        for i in tqdm(range(num_faces)):
            mat_body += "{} ".format(i)
        for i in tqdm(range(num_faces)):
            mat_body += "{} ".format(u[i, 0])
        for i in tqdm(range(num_faces)):
            mat_body += "{} ".format(u[i, 1])
        for i in tqdm(range(num_faces)):
            mat_body += "{} ".format(u[i, 2])
        for i in tqdm(range(num_faces)):
            mat_body += "{} ".format(v[i, 0])
        for i in tqdm(range(num_faces)):
            mat_body += "{} ".format(v[i, 1])
        for i in tqdm(range(num_faces)):
            mat_body += "{} ".format(v[i, 2])
        lines.append(mat_body)

        f.writelines(lines)


@click.command()
@click.option(
    "--inference-json-path",
    "-j",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to JSON model output, containing the predicted frame field.",
)
@click.option(
    "--source-obj-path",
    "-s",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to OBJ model of input triangle mesh.",
)
def convert_to_dmat(inference_json_path, source_obj_path):
    source_obj_file, inference_data_file = Path(source_obj_path), Path(
        inference_json_path
    )
    dmat_outpath = inference_json_path.replace("json", "dmat")

    assert source_obj_file.name.endswith("obj")
    assert inference_data_file.name.endswith("json")

    verts, faces = pp3d.read_mesh(str(source_obj_file))
    inference_data = json.load(open(inference_data_file))
    # pred_directions = inference_data["pred_directions"]
    for key in list(inference_data.keys()):
        inference_data[key] = np.array(inference_data[key])
    pred_directions = inference_data["pred_directions"]

    # main conversion logic:
    axis_x_list, axis_y_list, axis_n = compute_local_axes_triplanar(verts, faces)
    weight_yz, weight_xz, weight_xy = get_triplanar_weights(axis_n)

    frame_fields = get_frame_fields_global_coord(
        pred_directions, axis_x_list, axis_y_list
    )
    frame_fields = make_triplanes_consistent(frame_fields, axis_n)

    u = (
        frame_fields[:, 0:3] * weight_yz
        + frame_fields[:, 6:9] * weight_xz
        + frame_fields[:, 12:15] * weight_xy
    )
    v = (
        frame_fields[:, 3:6] * weight_yz
        + frame_fields[:, 9:12] * weight_xz
        + frame_fields[:, 15:18] * weight_xy
    )

    write_dmat_file(dmat_outpath, u, v)


if __name__ == "__main__":
    cli.add_command(convert_to_dmat)
    cli()
