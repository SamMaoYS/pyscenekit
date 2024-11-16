import os
from natsort import natsorted
from pyscenekit.scenekit3d.datasets.scannet.frame import ScanNetFrameDataset


class ScanNetDataset:
    """
    ScanNet: ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes

    Authors: Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias

    https://github.com/scannetpp/scannetpp

    @inproceedings{dai2017scannet,
        title={ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes},
        author={Dai, Angela and Chang, Angel X. and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
        booktitle = {Proc. Computer Vision and Pattern Recognition (CVPR), IEEE},
        year = {2017}
    }

    ScanNet Dataset Folder structure:
    data_dir
    ├── scan_id
        |-- scan_id.aggregation.json
        |-- scan_id.sens
        |-- scan_id.txt
        |-- scan_id_2d-instance-filt.zip
        |-- scan_id_2d-instance.zip
        |-- scan_id_2d-label-filt.zip
        |-- scan_id_2d-label.zip
        |-- scan_id_vh_clean.aggregation.json
        |-- scan_id_vh_clean.ply
        |-- scan_id_vh_clean.segs.json
        |-- scan_id_vh_clean_2.0.010000.segs.json
        |-- scan_id_vh_clean_2.labels.ply
        `-- scan_id_vh_clean_2.ply
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.scenes_ids = self.get_scenes_ids()
        self.current_scene_id = None
        self.frame_dataset = None
        self.mesh_dataset = None

    def _update(self):
        self.frame_dataset = ScanNetFrameDataset(
            self.current_scene_id, self.current_scene_path
        )

    def get_scenes_ids(self):
        folders = os.listdir(self.data_dir)
        return natsorted(
            [
                folder
                for folder in folders
                if os.path.isdir(os.path.join(self.data_dir, folder))
            ]
        )

    def set_scene_id_by_index(self, index: int):
        assert index < len(self.scenes_ids), "Index out of scenes ids range"
        self.set_scene_id(self.scenes_ids[index])
        self._update()

    def set_scene_id(self, scene_id: str):
        # check if scene_id is in scenes_ids
        if scene_id not in self.scenes_ids:
            raise ValueError(f"Scene {scene_id} not found in {self.data_dir}")
        self.current_scene_id = scene_id
        self._update()

    @property
    def current_scene_path(self):
        return os.path.join(self.data_dir, self.current_scene_id)
