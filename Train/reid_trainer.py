import os
import os.path as osp
import torchreid
from torchreid.reid.data.datasets.dataset import ImageDataset

class MyReidDataset(ImageDataset):
    dataset_dir = 'reid-data'

    def __init__(self, root='', **kwargs):
        # Resolve absolute paths
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # 1) collect all camera-folder names and person-folder names
        splits = ["train", "query", "gallery"]
        cam_names = set()
        pid_names = set()

        for split in splits:
            split_dir = osp.join(self.dataset_dir, split)
            if not osp.isdir(split_dir):
                raise ValueError(f"Expected '{split_dir}' to exist.")
            for cam in os.listdir(split_dir):
                cam_dir = osp.join(split_dir, cam)
                if not osp.isdir(cam_dir):
                    continue
                cam_names.add(cam)
                for pid in os.listdir(cam_dir):
                    pid_dir = osp.join(cam_dir, pid)
                    if osp.isdir(pid_dir):
                        pid_names.add(pid)

        # 2) build zero-based mappings
        cam_list = sorted(cam_names)
        pid_list = sorted(pid_names)
        cam2label = {cam: idx for idx, cam in enumerate(cam_list)}
        pid2label = {pid: idx for idx, pid in enumerate(pid_list)}

        # 3) helper to parse each split into (img_path, pid, camid)
        def parse_split(split):
            data = []
            split_dir = osp.join(self.dataset_dir, split)
            for cam in os.listdir(split_dir):
                cam_dir = osp.join(split_dir, cam)
                if not osp.isdir(cam_dir):
                    continue
                camid = cam2label[cam]
                for pid in os.listdir(cam_dir):
                    pid_dir = osp.join(cam_dir, pid)
                    if not osp.isdir(pid_dir):
                        continue
                    pidid = pid2label[pid]
                    # collect all image files under this pid folder
                    for fname in os.listdir(pid_dir):
                        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            continue
                        img_path = osp.join(pid_dir, fname)
                        data.append((img_path, pidid, camid))
            return data

        # 4) generate the three splits
        train = parse_split("train")
        query = parse_split("query")
        gallery = parse_split("gallery")

        # 5) pass to super
        super(MyReidDataset, self).__init__(train, query, gallery, **kwargs)


    # Move model to GPU if available
