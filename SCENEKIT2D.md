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

Currently, we support the following methods, change the `normal_estimation.method` to try different methods: `dsine`.

## TODO

- [ ] 🧩 Segmentation
- [ ] 📷 Camera Estimation
- [ ] 🎨 Visualization
