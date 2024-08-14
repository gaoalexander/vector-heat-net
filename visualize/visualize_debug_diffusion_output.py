import json

import numpy as np
import polyscope as ps
import potpourri3d as pp3d
import torch

from pathlib import Path

ps.init()


def visualize(verts, faces, input_vectors, diffused_vectors_1, diffused_vectors_2, diffused_vectors_3, axis_x, axis_y, axis_n):  # , tag):
    ps.register_surface_mesh(f"mesh", verts, faces, smooth_shade=False)
    ps.get_surface_mesh(f"mesh").add_vector_quantity(f"input_vectors",
                                                     input_vectors,
                                                     defined_on='vertices',
                                                     color=(1.0, 0.0, 1.0),
                                                     vectortype='ambient')

    ps.get_surface_mesh(f"mesh").add_vector_quantity(f"diffused_vectors_1",
                                                     diffused_vectors_1,
                                                     defined_on='vertices',
                                                     color=(0.0, 1.0, 1.0),
                                                     vectortype='ambient')
    ps.get_surface_mesh(f"mesh").add_vector_quantity(f"diffused_vectors_2",
                                                     diffused_vectors_2,
                                                     defined_on='vertices',
                                                     color=(0.0, 1.0, 1.0),
                                                     vectortype='ambient')
    ps.get_surface_mesh(f"mesh").add_vector_quantity(f"diffused_vectors_3",
                                                     diffused_vectors_3,
                                                     defined_on='vertices',
                                                     color=(0.0, 1.0, 1.0),
                                                     vectortype='ambient')

    ps.get_surface_mesh(f"mesh").add_vector_quantity(f"local_x",
                                                     axis_x,
                                                     defined_on='vertices',
                                                     color=(1.0, 0.0, 0.0))
    ps.get_surface_mesh(f"mesh").add_vector_quantity(f"local_y",
                                                     axis_y,
                                                     defined_on='vertices',
                                                     color=(0.0, 1.0, 0.0))
    ps.get_surface_mesh(f"mesh").add_vector_quantity(f"local_n",
                                                     axis_n,
                                                     defined_on='vertices',
                                                     color=(0.0, 0.0, 1.0))

    # ps.register_point_cloud("barycenters_pointcloud", barycenters)


def main():
    output_json_file_1 = "/Users/alexandergao/git/vector-diffusion-net/experiments/vector_diffusion/data/diffusion_debug_output/diffusion_time_0.001.json"
    output_json_file_2 = "/Users/alexandergao/git/vector-diffusion-net/experiments/vector_diffusion/data/diffusion_debug_output/diffusion_time_0.001_implicit.json"
    output_json_file_3 = "/Users/alexandergao/git/vector-diffusion-net/experiments/vector_diffusion/data/diffusion_debug_output/diffusion_time_0.1.json"

    mesh_file = "/Users/alexandergao/git/vector-diffusion-net/experiments/vector_diffusion/data/test001/meshes/train/spot_triangulated.obj"

    # print(output_json_file, mesh_file)

    verts, faces = pp3d.read_mesh(str(mesh_file))
    output_pt001 = json.load(open(output_json_file_1))
    output_pt01 = json.load(open(output_json_file_2))
    output_pt1 = json.load(open(output_json_file_3))


    preds = np.array(output_pt1["initial_vector_values"])

    targets_1 = np.array(output_pt001["diffused_vector_values"])
    targets_2 = np.array(output_pt01["diffused_vector_values"])
    targets_3 = np.array(output_pt1["diffused_vector_values"])

    axis_x = np.array(output_pt1["axis_x"])
    axis_y = np.array(output_pt1["axis_y"])
    axis_n = np.array(output_pt1["axis_n"])
    # tag = str(output_file).split('/')[-1].replace(".json", "").replace("output_", "")

    visualize(verts, faces, preds, targets_1, targets_2, targets_3, axis_x, axis_y, axis_n)  # , tag)

    ps.show()

if __name__ == "__main__":
    main()
