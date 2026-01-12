import os
import scipy.io
from PIL import Image
from torchvision import transforms, datasets 
from torch.utils.data import random_split, Dataset

def mnist(dataset_cfg: dict): 
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"] 
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  ds = datasets.MNIST(root='./data', train=True, transform=transform, download=True) 
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.MNIST(root='./data', train=False, transform=transform, download=True) 

  return train_ds, val_ds, test_ds

def fashion_mnist(dataset_cfg: dict):
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"] 
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  ds = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True) 
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True) 

  return train_ds, val_ds, test_ds

def cifar10(dataset_cfg: dict): 
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"]
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  ds = datasets.CIFAR10(root='./data/CIFAR10', train=True, transform=transform, download=True) 
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.CIFAR10(root='./data/CIFAR10', train=False, transform=transform, download=True) 

 
  return train_ds, val_ds, test_ds

def cifar100(dataset_cfg: dict): 
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"]
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  ds = datasets.CIFAR100(root='./data/CIFAR100', train=True, transform=transform, download=True) 
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.CIFAR100(root='./data/CIFAR100', train=False, transform=transform, download=True) 

 
  return train_ds, val_ds, test_ds

class StanfordCarsDataset(Dataset):
  def __init__(self, root, split="train", transform=None):
    self.root = root
    self.split = split
    self.transform = transform
    
    # Paths
    devkit = os.path.join(root, "devkit")
    if split == "train":
      self.img_dir = os.path.join(root, "cars_train")
      mat_path = os.path.join(devkit, "cars_train_annos.mat")
    else:
      self.img_dir = os.path.join(root, "cars_test")
      mat_path = os.path.join(devkit, "cars_test_annos.mat")
        
    self.samples = []
    if os.path.exists(mat_path):
      mat = scipy.io.loadmat(mat_path)
      annotations = mat["annotations"][0]
      for ann in annotations:
        fname = str(ann['fname'][0])
        if 'class' in ann.dtype.names:
          label = int(ann['class'][0, 0]) - 1
        else:
          label = -1
        self.samples.append((fname, label))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    fname, label = self.samples[idx]
    path = os.path.join(self.img_dir, fname)
    image = Image.open(path).convert("RGB")
    
    if self.transform:
      image = self.transform(image)
        
    return image, label

def cars(dataset_cfg: dict): 
  import kagglehub
  import shutil

  # download dataset 
  if not os.path.exists("data/cars"): 
    data_path = kagglehub.dataset_download("eduardo4jesus/stanford-cars-dataset")
    os.mkdir("data/cars")
    shutil.move(f"{data_path}/car_devkit/devkit", "./data/cars/")
    shutil.move(f"{data_path}/cars_train/cars_train", "./data/cars/")
    shutil.move(f"{data_path}/cars_test/cars_test", "./data/cars/")

  # transform and augment
  transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), # scales data to [0, 1]
  ])

  full_train_ds = StanfordCarsDataset(root='data/cars', split='train', transform=transform)

  # split
  train_split, val_split, _ = dataset_cfg["split"]
  total_split = train_split + val_split
  train_split = train_split / total_split
  val_split = val_split / total_split
  
  train_ds, val_ds = random_split(full_train_ds, [train_split, val_split])
  test_ds = StanfordCarsDataset(root='data/cars', split='test', transform=transform)
 
  return train_ds, val_ds, test_ds

def svhn(dataset_cfg: dict): 
  """
  There is also an "extra" split for SVHN
  """
  # transform and augment
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])

  # split
  train_split, val_split, _ = dataset_cfg["split"]
  total_split = train_split + val_split
  train_split, val_split = train_split / total_split, val_split / total_split
  ds = datasets.SVHN(root='./data/svhn', split='train', transform=transform, download=True) 
  train_ds, val_ds = random_split(ds, [train_split, val_split])
  test_ds = datasets.SVHN(root='./data/svhn', split='test', transform=transform, download=True) 

  return train_ds, val_ds, test_ds

def celebA(dataset_cfg: dict): 
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])

  train_ds = datasets.CelebA(root='./data', split='train', transform=transform, download=True)
  val_ds = datasets.CelebA(root='./data', split='valid', transform=transform, download=True)
  test_ds = datasets.CelebA(root='./data', split='test', transform=transform, download=True)

  return train_ds, val_ds, test_ds

def flowers102(dataset_cfg: dict):
  """
  Needs scipy to load target files
  """
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])


  train_ds = datasets.Flowers102(root='./data', split="train", transform=transform, download=True) 
  val_ds = datasets.Flowers102(root='./data', split="val", transform=transform, download=True) 
  test_ds = datasets.Flowers102(root='./data', split="test", transform=transform, download=True) 

  return train_ds, val_ds, test_ds

def makedir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def crop_and_split_images(dataset_root, output_root, val_split=0.2):
    """
    Crop images using bounding boxes and split into train/val/test sets.

    Args:
        dataset_root: Path to CUB_200_2011 dataset root
        output_root: Path to output directory (e.g., ./datasets/cub200_cropped/)
        val_split: Fraction of training data to use for validation.
    """

    # Define paths
    images_txt = os.path.join(dataset_root, 'images.txt')
    bboxes_txt = os.path.join(dataset_root, 'bounding_boxes.txt')
    split_txt = os.path.join(dataset_root, 'train_test_split.txt')
    images_dir = os.path.join(dataset_root, 'images')

    # Check if required files exist
    for file_path in [images_txt, bboxes_txt, split_txt]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Create output directories
    train_dir = os.path.join(output_root, 'train_cropped')
    val_dir = os.path.join(output_root, 'val_cropped')
    test_dir = os.path.join(output_root, 'test_cropped')
    makedir(train_dir)
    makedir(val_dir)
    makedir(test_dir)

    # Read images.txt: image_id image_path
    print("Reading images.txt...")
    image_paths = {}
    with open(images_txt, 'r') as f:
        for line in f:
            img_id, img_path = line.strip().split()
            image_paths[int(img_id)] = img_path

    # Read bounding_boxes.txt: image_id x y width height
    print("Reading bounding_boxes.txt...")
    bboxes = {}
    with open(bboxes_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_id = int(parts[0])
            x, y, width, height = map(float, parts[1:5])
            bboxes[img_id] = (x, y, width, height)

    # Read train_test_split.txt and create train/val split
    print("Reading train_test_split.txt and creating train/val split...")
    split_info = {}
    train_ids = []
    with open(split_txt, 'r') as f:
        for line in f:
            img_id, is_train = line.strip().split()
            split_info[int(img_id)] = int(is_train)
            if int(is_train) == 1:
                train_ids.append(int(img_id))

    import random
    random.shuffle(train_ids)
    val_size = int(len(train_ids) * val_split)
    val_ids = set(train_ids[:val_size])
    
    # Process each image
    print("Processing images...")
    num_train, num_val, num_test, num_errors = 0, 0, 0, 0

    for img_id in sorted(image_paths.keys()):
        img_path = image_paths[img_id]
        bbox = bboxes[img_id]
        is_train = split_info[img_id]

        # Construct full image path
        full_img_path = os.path.join(images_dir, img_path)

        if not os.path.exists(full_img_path):
            print(f"Warning: Image not found: {full_img_path}")
            num_errors += 1
            continue

        try:
            # Load image
            img = Image.open(full_img_path)
            img = img.convert('RGB')

            # Crop using bounding box
            x, y, width, height = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + width), int(y + height)

            # Ensure bbox is within image bounds
            img_width, img_height = img.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_width, x2), min(img_height, y2)

            cropped_img = img.crop((x1, y1, x2, y2))

            # Determine output directory and create class subdirectory
            class_name = img_path.split('/')[0]

            if is_train == 1:
                if img_id in val_ids:
                    output_dir = os.path.join(val_dir, class_name)
                    num_val += 1
                else:
                    output_dir = os.path.join(train_dir, class_name)
                    num_train += 1
            else:
                output_dir = os.path.join(test_dir, class_name)
                num_test += 1

            makedir(output_dir)

            # Save cropped image
            img_filename = os.path.basename(img_path)
            output_path = os.path.join(output_dir, img_filename)
            cropped_img.save(output_path)

            if (num_train + num_val + num_test) % 500 == 0:
                print(f"Processed {num_train + num_val + num_test} images...")

        except Exception as e:
            print(f"Error processing {full_img_path}: {e}")
            num_errors += 1
            continue

    print("\n" + "="*60)
    print(f"Processing complete!")
    print(f"Training images: {num_train}")
    print(f"Validation images: {num_val}")
    print(f"Test images: {num_test}")
    print(f"Errors: {num_errors}")
    print(f"Train output: {train_dir}")
    print(f"Val output: {val_dir}")
    print(f"Test output: {test_dir}")
    print("="*60)

def cub200(dataset_cfg: dict): 
  import kagglehub
  import shutil
  import Augmentor

  # download dataset 
  if not os.path.exists("data/CUB_200_2011"): 
    data_path = kagglehub.dataset_download("wenewone/cub2002011")
    src_path = os.path.join(data_path, "CUB_200_2011")
    target_path = "./data/CUB_200_2011"
    
    # Ensure clean target
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
        
    if os.path.exists(src_path):
        print(f"Copying dataset from {src_path} to {target_path}...")
        shutil.copytree(src_path, target_path)
    elif os.path.exists(os.path.join(data_path, "images")):
         print(f"Adapting flat dataset structure from {data_path}...")
         shutil.copytree(data_path, target_path)
    else:
         raise FileNotFoundError(f"Expected 'CUB_200_2011' folder or 'images' in {data_path}, found: {os.listdir(data_path)}")

  # crop and make train/validation/test split
  if not os.path.exists("data/cub200_cropped"):
    crop_and_split_images(dataset_root="./data/CUB_200_2011", output_root="./data/cub200_cropped")

  # augment train set
  if not os.path.exists("data/cub200_cropped/train_cropped_augmented"): 
    dir = os.path.abspath('./data/cub200_cropped/train_cropped/')
    target_dir = os.path.abspath('./data/cub200_cropped/train_cropped_augmented/')

    makedir(target_dir)

    folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
    target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

    for i in range(0, len(folders)):
      fd = folders[i]
      tfd = target_folders[i]
      # rotation
      p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd) 
      p.rotate(probability=1, max_left_rotation=10, max_right_rotation=10)
      p.flip_left_right(probability=0.5)
      for i in range(10):
        p.process()
      del p
      # skew
      p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
      p.skew(probability=1, magnitude=0.2)  # max 45 degrees
      p.flip_left_right(probability=0.5)
      for i in range(10):
        p.process()
      del p
      # shear
      p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
      p.shear(probability=1, max_shear_left=10, max_shear_right=10)
      p.flip_left_right(probability=0.5)
      for i in range(10):
        p.process()
      del p
      # random_distortion
      #p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
      #p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
      #p.flip_left_right(probability=0.5)
      #for i in range(10):
      #    p.process()
      #del p

  normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                   std=(0.229, 0.224, 0.225))

  train_ds = datasets.ImageFolder(
      "./data/cub200_cropped/train_cropped_augmented/", 
      transforms.Compose([
          transforms.Resize(size=(224, 224)),
          transforms.ToTensor(),
          normalize,
      ]))
  
  train_push_ds = datasets.ImageFolder(
      "./data/cub200_cropped/train_cropped/", 
      transforms.Compose([
          transforms.Resize(size=(224, 224)),
          transforms.ToTensor(),
      ]))

  val_ds = datasets.ImageFolder(
      "./data/cub200_cropped/val_cropped/", 
      transforms.Compose([
          transforms.Resize(size=(224, 224)),
          transforms.ToTensor(),
          normalize,
      ]))

  test_ds = datasets.ImageFolder(
      "./data/cub200_cropped/test_cropped/", 
      transforms.Compose([
          transforms.Resize(size=(224, 224)),
          transforms.ToTensor(),
          normalize,
      ]))

  return train_ds, train_push_ds, val_ds, test_ds

