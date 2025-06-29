import os
import os.path as osp
import torch
import torchreid
from torchreid.reid.data.datasets.dataset import ImageDataset
from torchreid.reid.models import build_model


class MyReidDataset(ImageDataset):
    """
    Custom ReID dataset class for football player re-identification.
    Inherits from torchreid's ImageDataset class.
    """
    dataset_dir = 'drive/MyDrive/stealth mode/reid-data_mine'

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

        def check_pid_consistency(query, gallery):
            query_pids = set([pid for _, pid, _ in query])
            gallery_pids = set([pid for _, pid, _ in gallery])

            missing = query_pids - gallery_pids
            if missing:
                print("🚫 Missing PIDs in gallery:", missing)
            else:
                print("✅ All query PIDs are present in gallery!")

        # Call it like this
        check_pid_consistency(query, gallery)

        # 5) pass to super
        super(MyReidDataset, self).__init__(train, query, gallery, **kwargs)


class ReidTrainer:
    """
    ReID Trainer class for training person re-identification models.
    
    Example usage:
    ```python
    # Initialize the trainer
    trainer = ReidTrainer(
        root_dir='',
        dataset_dir='reid-data',
        model_name='osnet_x1_0',
        save_dir='og/osnet',
        max_epochs=1
    )
    
    # Train the model
    trainer.train()
    ```
    """
    
    def __init__(self, root_dir='', dataset_dir='reid-data', model_name='osnet_x1_0', 
                 save_dir='og/osnet', max_epochs=1, eval_freq=1, print_freq=2):
        """
        Initialize the ReID trainer.
        
        Args:
            root_dir (str): Root directory for the dataset
            dataset_dir (str): Directory containing the dataset
            model_name (str): Name of the model to use
            save_dir (str): Directory to save the trained model
            max_epochs (int): Maximum number of training epochs
            eval_freq (int): Evaluation frequency
            print_freq (int): Print frequency
        """
        self.root_dir = root_dir
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.save_dir = save_dir
        self.max_epochs = max_epochs
        self.eval_freq = eval_freq
        self.print_freq = print_freq
        
        # Initialize components
        self.datamanager = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.engine = None
        
    def setup_data_manager(self):
        """Setup the data manager for training."""
        self.datamanager = torchreid.data.ImageDataManager(
            root=self.root_dir,
            sources='MY-reid-dataset1',
            transforms=None,
            workers=2,
            height=256,
            width=256
        )
        
    def setup_model(self):
        """Setup the model, optimizer, and scheduler."""
        self.model = build_model(
            name=self.model_name,
            num_classes=self.datamanager.num_train_pids,
            loss='softmax',
            pretrained=True,
            use_gpu=torch.cuda.is_available()
        )
        
        self.optimizer = torchreid.optim.build_optimizer(
            self.model,
            optim='adam',
            lr=0.0003
        )
        
        self.scheduler = torchreid.optim.build_lr_scheduler(
            self.optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        
    def setup_engine(self):
        """Setup the training engine."""
        self.engine = torchreid.reid.engine.image.ImageSoftmaxEngine(
            self.datamanager,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            label_smooth=True,
            use_gpu=torch.cuda.is_available()
        )
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
    def train(self):
        """Train the model."""
        # Setup all components
        self.setup_data_manager()
        self.setup_model()
        self.setup_engine()
        
        # Run training
        self.engine.run(
            save_dir=self.save_dir,
            max_epoch=self.max_epochs,
            eval_freq=self.eval_freq,
            print_freq=self.print_freq,
            test_only=False
        )


# Example usage:
if __name__ == "__main__":
    """
    Example of how to use the ReidTrainer class:
    
    # Create trainer instance
    trainer = ReidTrainer(
        root_dir='',                    # Root directory
        dataset_dir='reid-data',        # Dataset directory
        model_name='osnet_x1_0',        # Model architecture
        save_dir='og/osnet',            # Save directory
        max_epochs=1,                   # Number of epochs
        eval_freq=1,                    # Evaluation frequency
        print_freq=2                    # Print frequency
    )
    
    # Start training
    trainer.train()
    """
    
    # Initialize trainer
    trainer = ReidTrainer()
    
    # Train the model
    trainer.train() 