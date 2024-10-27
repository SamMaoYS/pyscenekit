import os
from pyscenekit.scenekit3d.datasets.scannetpp.dlsr import ScanNetPPDLSRDataset


class ScanNetPPDataset:
    """
    ScanNetPP Dataset Folder structure:
    data_dir
    ├── scene_id
        ├── dslr
        ├── iphone
        └── scans
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.scenes_ids = self.get_scenes_ids()
        self.current_scene_id = self.scenes_ids[0]
        self.dlsr_dataset = ScanNetPPDLSRDataset(self.current_scene_dslr_path)

    def _update(self):
        self.dlsr_dataset = ScanNetPPDLSRDataset(self.current_scene_dslr_path)

    def get_scenes_ids(self):
        folders = os.listdir(self.data_dir)
        return [
            folder
            for folder in folders
            if os.path.isdir(os.path.join(self.data_dir, folder))
        ]

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

    @property
    def current_scene_dslr_path(self):
        return os.path.join(self.current_scene_path, "dslr")

    @property
    def current_scene_iphone_path(self):
        return os.path.join(self.current_scene_path, "iphone")

    @property
    def current_scene_scans_path(self):
        return os.path.join(self.current_scene_path, "scans")
