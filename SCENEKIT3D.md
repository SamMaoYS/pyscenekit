# SceneKit3D

SceneKit3D is a module focusing on 3D scene processing, analysis and visualization. Checkout the main config file in [configs](./configs/scenekit3d.yaml).

## Dataset

### ScanNet++

[ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) is a large scale dataset with 450+ 3D indoor scenes containing sub-millimeter resolution laser scans, registered 33-megapixel DSLR images, and commodity RGB-D streams from iPhone.

Example usage:

```bash
python examples/scannetpp_dataset.py output=outputs/scannetpp
```
Currently, we support the DLSR images undistortion.

## Multi-view Reconstruction

Multi-view reconstruction takes multiple input images and generates coarse/dense point clouds of the scene. Some methods may also estimate camera poses during reconstruction.

Example usage:

```bash
python examples/multiview_reconstruction.py visualization.interactive=true multiview_reconstruction.method=dust3r input="examples/data/scannetpp_6b40d1a939_*.JPG" output=outputs/reconstruction.pth
```
Currently, we support the following methods, change the `multiview_reconstruction.method` to try different methods: `dust3r`.


## TODO

- [ ] 🏗️ 3D Reconstruction
- [ ] 🧠 3D Scene Understanding
- [ ] 🖼️ 3D Scene Visualization
- [ ] 🏠 3D Scene Datasets
