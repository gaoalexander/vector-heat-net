# An Intrinsic Vector Heat Network (ICML 2024)
[Alexander Gao](https://gaoalexander.github.io/), [Maurice Chu](https://www.linkedin.com/in/maurice-chu-1ab3731a/), [Mubbasir Kapadia](https://scholar.google.com/citations?user=xhkzmycAAAAJ&hl=en/), [Ming C. Lin](https://www.cs.umd.edu/~lin/), [Hsueh-Ti Derek Liu](https://www.dgp.toronto.edu/~hsuehtil/)
### [[Paper]](https://arxiv.org/abs/2406.09648) [[Project Page]](https://gaoalexander.github.io/vector-heat-network/)

#### Acknowledgement: This is a research codebase partially based on the original [DiffusionNet](https://github.com/nmwsharp/diffusion-net) and [DeltaConv](https://github.com/rubenwiersma/deltaconv) implementations.  We thank the authors for their works and open source code, which we have been able to build our own work upon.

# Setup
PyTorch training code will run on CUDA backend if it is available, otherwise will revert to CPU.  MPS backend on MacOS is not available, as sparse tensor operations are not officially supported.

### Setup python environment
Create a new conda environment and install python dependencies:
```
conda create -n vectorheatnet python=3.9
conda activate vectorheatnet
pip install -r requirements.txt
```

### Install C++ Dependencies (only required for Quad Meshing)
Install [LibIGL](https://github.com/libigl/libigl) and [LibQEx](https://github.com/hcebke/libQEx), which are used to produce a quadrilateral mesh from an estimated vector field (cross field).

# Run
### Train
To get started, train a Vector Heat Network on provided example data:
```
python experiments/quad_meshing/train.py --dataset_path experiments/quad_meshing/data/example_quadwild
```
Model checkpoints and test output are saved in the `experiments/quad_meshing/output` directory.
### Inference
```
python experiments/quad_meshing/inference.py --pretrain-path <path/to/output/run/dir>
```

### Extract quadrilateral meshes from inferred vector field
See the [cpp_mesh_extraction](experiments/quad_meshing/cpp_mesh_extraction/README.md) module for additional information on extracting a quadrilateral mesh from the predicted cross field. 

# Data
### Creating a custom dataset for quadrilateral meshing
```
python experiments/quad_meshing/dataset/preprocess_custom_quad_dataset.py --source_quad_mesh_dir experiments/quad_meshing/data/example_custom_dataset/quad --output_dir experiments/quad_meshing/data/example_custom_dataset/preprocessed/train
```
# Visualize results

# Citation
___

```
@inproceedings{gao2024intrinsic,
    title       = {An Intrinsic Vector Heat Network},
    author      = {Gao, Alexander and Chu, Maurice and Kapadia, Mubbasir and Lin, Ming and Liu, Hsueh-Ti Derek},
    booktitle   = {Forty-first International Conference on Machine Learning},
    year        = {2024}
}
```

