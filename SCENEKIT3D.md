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

## TODO

- [ ] 🏗️ 3D Reconstruction
- [ ] 🧠 3D Scene Understanding
- [ ] 🖼️ 3D Scene Visualization
- [ ] 🏠 3D Scene Datasets
