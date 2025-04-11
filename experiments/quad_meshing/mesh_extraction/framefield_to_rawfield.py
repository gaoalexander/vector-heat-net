import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import polyscope as ps
import potpourri3d as pp3d
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))  # add the path to the DiffusionNet src
from experiments.quad_meshing.mesh_extraction.interpolate_vertices_to_faces import interpolate_vector_field_v_to_f
from experiments.quad_meshing.mesh_extraction.visualize_mesh import visualize_mesh


def write_dmat_file(dmat_outpath, u, v):
    with open(dmat_outpath, "a") as f:
        num_faces = u.shape[0]
        lines = ["{} {}\n".format(7, num_faces)]
        mat_body = ""
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


def convert_to_dmat(inference_json_path, source_obj_path):
    source_obj_file, inference_data_file = Path(source_obj_path), Path(
        inference_json_path
    )
    dmat_outpath = inference_json_path.replace("json", "dmat")

    assert source_obj_file.name.endswith("obj")
    assert inference_data_file.name.endswith("json")

    verts, faces = pp3d.read_mesh(str(source_obj_file))
    inference_data = json.load(open(inference_data_file))
    # # pred_directions = inference_data["pred_directions"]
    for key in list(inference_data.keys()):
        inference_data[key] = np.array(inference_data[key])
    pred_directions = inference_data["preds_local"]
    pred_directions = np.concatenate((pred_directions[::2], pred_directions[1:][::2]), axis=1)

    # main conversion logic:
    pred_directions_faces = interpolate_vector_field_v_to_f(
        verts,
        faces,
        pred_directions,
        inference_data["axis_x_verts"],
        inference_data["axis_x_faces"]
    )


    u = pred_directions_faces.real[:, None] * inference_data["axis_x_faces"] + \
        pred_directions_faces.imag[:, None] * inference_data["axis_y_faces"]
    v = (pred_directions_faces * np.exp(np.pi / 2 * (0 + 1j))).real[:, None] * inference_data["axis_x_faces"] + \
        (pred_directions_faces * np.exp(np.pi / 2 * (0 + 1j))).imag[:, None] * inference_data["axis_y_faces"]

    # ps.init()
    # visualize_mesh(verts, faces, vectors=inference_data["preds"])
    # visualize_mesh(verts, faces, vectors=u, defined_on="faces", name="u")
    # visualize_mesh(verts, faces, vectors=v, defined_on="faces", name="v")
    # ps.show()

    write_dmat_file(dmat_outpath, u, v)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_json_path",
                        type=str,
                        help="Path to JSON model output, containing the predicted frame field.")
    parser.add_argument("--source_obj_path",
                        type=str,
                        help="Path to OBJ model of input triangle mesh.",
                        )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    convert_to_dmat(args.inference_json_path, args.source_obj_path)
