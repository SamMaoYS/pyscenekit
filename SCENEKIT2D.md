# SceneKit2D

SceneKit2D is a module within PySceneKit that focuses on 2D scene processing and analysis. It provides a set of tools and algorithms for working with 2D images, particularly in the context of indoor scenes. Checkout the main config file in [configs](./configs/scenekit2d.yaml).

## Depth Estimation

Utilize state-of-the-art models like Depth Anything V2 to estimate depth from single images.

Example usage:

```bash
python examples/depth_estimation.py depth_estimation.method=depth_anything_v2 input=examples/data/bedroom_fluxdev.jpg
output=outputs/bedroom_fluxdev_depth.jpg
```
Currently, we support the following methods, change the `depth_estimation.method` to try different methods: `midas`, `depth_anything_v2`, `depth_pro`, `lotus_depth`.

## Normal Estimation

Implement advanced techniques such as DSINE for accurate surface normal prediction.

Example usage:

```bash
python examples/normal_estimation.py normal_estimation.method=dsine output=outputs/bedroom_fluxdev_normal.jpg
```

Currently, we support the following methods, change the `normal_estimation.method` to try different methods: `dsine`, `lotus_normal`.

## Semantic Segmentation

Semantic segmentation is the task of classifying each pixel in an image into a set of predefined categories.

Example usage:

```bash
python examples/image_segmentation.py image_segmentation.method=upernet input=examples/data/bedroom_fluxdev.jpg
output=outputs/bedroom_fluxdev_semantic.jpg
```

Currently, we support the following methods, change the `image_segmentation.method` to try different methods: `upernet`.

## Camera Estimation

Camera estimation is the task of estimating the camera parameters from a single image, such as camera intrinsics, extrinsics, world gravity direction, etc.

Example usage:

```bash
python examples/camera_estimation.py camera_estimation.method=geo_calib input=examples/data/bedroom_fluxdev.jpg
output=outputs/bedroom_fluxdev_camera.pth
```

Currently, we support the following methods, change the `camera_estimation.method` to try different methods: `geo_calib`, `vp_prior_gravity`, `vp_houghtransform_gaussiansphere`.

> **Note**: The vanishing point estimation methods (`vp_prior_gravity`, `vp_houghtransform_gaussiansphere`) currently require additional C++ dependencies and compilation. I am working on optimizing the code and simplifying the installation process for these methods in future releases.
Checkout the [INSTALLATION.md](./INSTALLATION.md) for more details.

## TODO

- [ ] 🧩 Segmentation
- [ ] 🎨 Visualization
