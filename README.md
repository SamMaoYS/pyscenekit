# PySceneKit

PySceneKit is an open-source Python library designed for common scene processing and visualization tasks. Whether you're working with 2D or 3D scenes, PySceneKit provides a comprehensive toolkit to help you manipulate, analyze, and visualize your data with ease.

## Introduction

Welcome to PySceneKit! This project is fueled by my passion for scene understanding, particularly in indoor environments. Frustrated by the lack of user-friendly tools for processing images and 3D indoor scenes, I often found myself rewriting code for different datasets.

PySceneKit aims to simplify scene understanding by providing an intuitive toolkit that incorporates both state-of-the-art techniques and classic methods tailored for indoor scenes. I warmly welcome contributions from fellow enthusiasts and researchers to help make this project a valuable resource for everyone!


## Installation

```bash
conda create -n pyscenekit python=3.10 -y
conda activate pyscenekit
pip install -r requirements.txt
pip install -e . # will support pip install pyscenekit in the future
```

## Usage

SceneKit2D is a module within PySceneKit that focuses on 2D scene processing and analysis. It provides a set of tools and algorithms for working with 2D images, particularly in the context of indoor scenes. Checkout the key features in [SceneKit2D](./SCENEKIT2D.md).

SceneKit3D is a module focusing on 3D scene processing and analysis. Checkout the key features in [SceneKit3D](./SCENEKIT3D.md).


## Progress

- [x] 🌟 **Kicking Off the Adventure**: Initial setup complete!
- [ ] 💡 **Bringing Ideas to Life**: Core functionalities in the works.
- [ ] 🎨 **Crafting Beautiful Scenes**: Building basic scene processing features.
- [ ] 🔍 **Visual Wonderland**: Adding stunning visualization tools.
- [ ] 📚 **Sharing the Love**: Writing documentation and creating examples.
- [ ] 🔧 **Polishing the Gem**: Testing and optimizing for the best performance.
- [ ] 🚀 **Launch Countdown**: Preparing for the exciting release of v1.0!

## Acknowledgments

PySceneKit would not be possible without the incredible work of various open-source projects and libraries that have paved the way for scene processing and visualization. For a detailed list of acknowledgments, please see the [ACKNOWLEDGMENTS.md](./ACKNOWLEDGMENTS.md) file.

## Citation

If you find PySceneKit useful in your research, please consider citing the project:

```bibtex
@misc{mao2024pyscenekit,
    title={PySceneKit GitHub Repository},
    author={Mao, Yongsen},
    year={2024}
}
```
