# Quadrilateral mesh extraction

### Setup
The following process depends on installing [LibIGL](https://github.com/libigl/libigl) and [LibQEx](https://github.com/hcebke/libQEx).

Once those dependencies have been installed, compile `cpp/01_parametrize.cpp` and `cpp/02_extract_quads.cpp` into binaries.

### Usage:
We first need to convert the inferred vector field (`.json`) to a format that is easily readable by LibIGL (`.dmat`):
```
python experiments/quad_meshing/dataset/framefield_to_rawfield.py <path/to/inference_output.json> <path/to/source_mesh.obj>
```

Then, guided by the vector field (which is now in `.dmat` format), we use IGL to deform and compute a 2D parameterization of the input mesh.  This step will output a file in the same directory as the input mesh, with extension `<meshname>_parameterized.obj`:
```
./01_parameterize <path/to/source_mesh.obj> <path/to/inference_output.dmat>
```

Finally, using the parameterization, we extract the final quadrilateral mesh:
```
./02_extract_quads <path/to/source_mesh_parameterized.obj> <path/to/source_mesh_quads.obj
```