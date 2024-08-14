import os

import bpy
# import click


# @click.group()
# def cli():
#     """Command-line tools for retopo experiments"""
#     pass
#
#
# @click.command()
# @click.option(
#     "--source-obj-dir",
#     "-s",
#     type=click.Path(exists=True, dir_okay=False),
#     required=True,
#     help="Path to directory containing source OBJ files (these are raw quad-dominant artist-defined topologies).",
# )
# @click.option(
#     "--output-dir",
#     "-s",
#     type=click.Path(exists=True, dir_okay=False),
#     required=True,
#     help="Path to directory for storing output OBJ files adter CC subdivision has been applied.",
# )
# @click.option(
#     "--num-iterations",
#     "-n",
#     type=int,
#     required=True,
#     default=1,
#     help="Number of Catmull-Clark subdivisions to apply.",
# )
def batch_apply_cc_subdivision(source_obj_dir, output_dir, num_iterations):
    os.makedirs(output_dir, exist_ok=True)
    done = set(os.listdir(output_dir))

    for filename in os.listdir(source_obj_dir):
        filepath = os.path.join(source_obj_dir, filename)
        if (
            filepath.endswith(".obj")
            and not filepath.endswith("shell.obj")
            and filename not in done
        ):
            # Load the OBJ file
            mesh = bpy.ops.import_scene.obj(filepath=filepath)

            # Select the imported object
            obj = bpy.context.selected_objects[0]
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

            modifier_name = "SubSurf"
            if obj.modifiers.get(modifier_name) is None:
                # Add Subsurface Modifier if not present
                obj.modifiers.new(name=modifier_name, type="SUBSURF")

            # Set the number of subdivisions
            obj.modifiers[modifier_name].levels = num_iterations
            bpy.ops.object.multires_subdivide(
                modifier=modifier_name, mode="CATMULL_CLARK"
            )

            # Save the resulting mesh as a new OBJ file
            result_obj_path = os.path.join(
                output_dir, filename.replace(".obj", "_cc{}.obj".format(num_iterations))
            )
            bpy.ops.export_scene.obj(
                filepath=result_obj_path,
                use_selection=True,
                use_materials=False,
                use_mesh_modifiers=True,
                use_normals=False,
                use_uvs=False,
            )

            print("Subdivision completed. Result saved as:", result_obj_path)


if __name__ == "__main__":
    """
    Example on how to run this script via CLI:
        /Applications/Blender.app/Contents/MacOS/blender --background --python bpy_cc_subdivide.py -- --click_args...
    """
    batch_apply_cc_subdivision(source_obj_dir="/Users/alexandergao/git/vector-diffusion-net/experiments/quad_meshing/data/test_triangulations/testdata",
                               output_dir="/Users/alexandergao/git/vector-diffusion-net/experiments/quad_meshing/data/test_triangulations/testdata_cc",
                               num_iterations=1)
    # cli.add_command(batch_apply_cc_subdivision)
    # cli()
