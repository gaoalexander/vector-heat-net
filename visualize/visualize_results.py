import json

import numpy as np
import polyscope as ps
import potpourri3d as pp3d
import torch

from pathlib import Path

ps.init()

def visualize(verts, faces, preds, targets, axis_x, axis_y, axis_n, tag):
    ps.register_surface_mesh(f"mesh_{tag}", verts, faces, smooth_shade=False)
    ps.get_surface_mesh(f"mesh_{tag}").add_vector_quantity(f"preds_{tag}",
                                                             preds,
                                                             defined_on='vertices',
                                                             color=(1.0, 0.0, 1.0))
    ps.get_surface_mesh(f"mesh_{tag}").add_vector_quantity(f"targets_{tag}",
                                                             targets,
                                                             defined_on='vertices',
                                                             color=(0.0, 1.0, 1.0))
    ps.get_surface_mesh(f"mesh_{tag}").add_vector_quantity(f"local_x_{tag}",
                                                             axis_x,
                                                             defined_on='vertices',
                                                             color=(1.0, 0.0, 0.0))
    ps.get_surface_mesh(f"mesh_{tag}").add_vector_quantity(f"local_y_{tag}",
                                                             axis_y,
                                                             defined_on='vertices',
                                                             color=(0.0, 1.0, 0.0))
    ps.get_surface_mesh(f"mesh_{tag}").add_vector_quantity(f"local_n_{tag}",
                                                             axis_n,
                                                             defined_on='vertices',
                                                             color=(0.0, 0.0, 1.0))

    # ps.register_point_cloud("barycenters_pointcloud", barycenters)

def main():
    output_train_path = "/Users/alexandergao/git/vector-diffusion-net/experiments/vector_diffusion/data/test001/output/output_train.json"
    mesh_train_path = "/Users/alexandergao/git/vector-diffusion-net/experiments/vector_diffusion/data/test001/meshes/train/spot_triangulated.obj"
    
    output_test_path = "/Users/alexandergao/git/vector-diffusion-net/experiments/vector_diffusion/data/test001/output/output_test.json"
    mesh_test_path = "/Users/alexandergao/git/vector-diffusion-net/experiments/vector_diffusion/data/test001/meshes/test/spot_triangulated_rotated.obj"

    output_files = [Path(output_train_path), Path(output_test_path)]
    mesh_files = [Path(mesh_train_path), Path(mesh_test_path)]

    for output_file, mesh_file in zip(output_files, mesh_files):
        print(output_file, mesh_file)

        assert mesh_file.name.endswith("obj")
        assert output_file.name.endswith("json")

        verts, faces = pp3d.read_mesh(str(mesh_file))
        output = json.load(open(output_file))

        preds = np.array(output["preds"])
        targets = np.array(output["targets"])
        axis_x = np.array(output["axis_x"])
        axis_y = np.array(output["axis_y"])
        axis_n = np.array(output["axis_n"])
        tag = str(output_file).split('/')[-1].replace(".json", "").replace("output_", "")

        visualize(verts, faces, preds, targets, axis_x, axis_y, axis_n, tag)

    ps.show()


if __name__ == "__main__":
    main()