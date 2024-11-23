import argparse
import csv
import glob
import os

from PIL import Image
import cv2
from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
import torch
from pathlib import Path

import library.train_util as train_util

# from wd14 tagger
IMAGE_SIZE = 448


DEFAULT_WD14_TAGGER_REPO = 'SmilingWolf/wd-v1-4-convnext-tagger-v2'
FILES = ["keras_metadata.pb", "saved_model.pb", "selected_tags.csv"]
SUB_DIR = "variables"
SUB_DIR_FILES = ["variables.data-00000-of-00001", "variables.index"]
CSV_FILE = FILES[-1]

def preprocess_image(image):
    image = np.array(image)
    image = image[:, :, ::-1]                         # RGB->BGR

    # pad to square
    size = max(image.shape[0:2])
    pad_x = size - image.shape[1]
    pad_y = size - image.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    image = np.pad(image, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode='constant', constant_values=255)

    interp = cv2.INTER_AREA if size > IMAGE_SIZE else cv2.INTER_LANCZOS4
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=interp)

    image = image.astype(np.float32)
    return image

class ImageLoadingPrepDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = str(self.images[idx])

        try:
            image = Image.open(img_path).convert("RGB")
            image = preprocess_image(image)
            tensor = torch.tensor(image)
        except Exception as e:
            print(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor, img_path)
  
def collate_fn_remove_corrupted(batch):
    """Collate function that allows to remove corrupted examples in the
    dataloader. It expects that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are removed.
    """
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    return batch

def main(args):

    if not os.path.exists(args.model_dir) or args.force_download:
        print(f"downloading wd14 tagger model from hf_hub. id: {args.repo_id}")
        for file in FILES:
            hf_hub_download(args.repo_id, file, cache_dir=args.model_dir, force_download=True, force_filename=file)
        for file in SUB_DIR_FILES:
            hf_hub_download(args.repo_id, file, subfolder=SUB_DIR, cache_dir=os.path.join(
                args.model_dir, SUB_DIR), force_download=True, force_filename=file)
    else:
        print("using existing wd14 tagger model")

    model = load_model(args.model_dir)


    with open(os.path.join(args.model_dir, CSV_FILE), "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        l = [row for row in reader]
        header = l[0]
        rows = l[1:]
    assert header[0] == 'tag_id' and header[1] == 'name' and header[2] == 'category', f"unexpected csv format: {header}"

    general_tags = [row[1] for row in rows[1:] if row[2] == '0']
    character_tags = [row[1] for row in rows[1:] if row[2] == '4']

    
    train_data_dir = Path(args.train_data_dir)
    image_paths = train_util.glob_images_pathlib(train_data_dir, args.recursive)
    print(f"found {len(image_paths)} images.")

    tag_freq = {}

    undesired_tags = set(args.undesired_tags.split(','))

    def run_batch(path_imgs):
        imgs = np.array([im for _, im in path_imgs])

        probs = model(imgs, training=False)
        probs = probs.numpy()

        for (image_path, _), prob in zip(path_imgs, probs):

            combined_tags = []
            general_tag_text = ""
            character_tag_text = ""
            for i, p in enumerate(prob[4:]):
                if i < len(general_tags) and p >= args.general_threshold:
                    tag_name = general_tags[i].replace('_', ' ') if args.remove_underscore else general_tags[i]
                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        general_tag_text += ", " + tag_name
                        combined_tags.append(tag_name)
                elif i >= len(general_tags) and p >= args.character_threshold:
                    tag_name = character_tags[i - len(general_tags)].replace('_', ' ') if args.remove_underscore else character_tags[i - len(general_tags)]
                    if tag_name not in undesired_tags:
                        tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1
                        character_tag_text += ", " + tag_name
                        combined_tags.append(tag_name)

            if len(general_tag_text) > 0:
                general_tag_text = general_tag_text[2:]

            if len(character_tag_text) > 0:
                character_tag_text = character_tag_text[2:]

            tag_text = ', '.join(combined_tags)
            
            with open(os.path.splitext(image_path)[0] + args.caption_extension, "wt", encoding='utf-8') as f:
                f.write(tag_text + '\n')
                if args.debug:
                    print(f"\n{image_path}:\n  Character tags: {character_tag_text}\n  General tags: {general_tag_text}")

    if args.max_data_loader_n_workers is not None:
        dataset = ImageLoadingPrepDataset(image_paths)
        data = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.max_data_loader_n_workers, collate_fn=collate_fn_remove_corrupted, drop_last=False)
    else:
        data = [[(None, ip)] for ip in image_paths]

    b_imgs = []
    for data_entry in tqdm(data, smoothing=0.0):
        for data in data_entry:
            if data is None:
                continue

            image, image_path = data
            if image is not None:
                image = image.detach().numpy()
            else:
                try:
                    image = Image.open(image_path)
                    if image.mode != 'RGB':
                        image = image.convert("RGB")
                    image = preprocess_image(image)
                except Exception as e:
                    print(f"Could not load image path : {image_path}, error: {e}")
                    continue
            b_imgs.append((image_path, image))

            if len(b_imgs) >= args.batch_size:
                b_imgs = [(str(image_path), image) for image_path, image in b_imgs]
                run_batch(b_imgs)
                b_imgs.clear()

    if len(b_imgs) > 0:
        b_imgs = [(str(image_path), image) for image_path, image in b_imgs]
        run_batch(b_imgs)

    if args.frequency_tags:
        sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
        print("\nTag frequencies:")
        for tag, freq in sorted_tags:
            print(f"{tag}: {freq}")

    print("done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, default="", help="Directory for training images")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_WD14_TAGGER_REPO, help="Repository ID of the wd14 tagger on Hugging Face")
    parser.add_argument("--model_dir", type=str, default="", help="Directory to store the wd14 tagger model")
    parser.add_argument("--force_download", action='store_true', help="Force re-download of the wd14 tagger model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=2, help="Number of worker threads to read images with DataLoader (faster)")
    parser.add_argument("--caption_extention", type=str, default=None, help="Extension of caption files (for backward compatibility)")
    parser.add_argument("--caption_extension", type=str, default=".txt", help="Extension of caption files")
    parser.add_argument("--general_threshold", type=float, default=0.7, help="Confidence threshold for adding general category labels")
    parser.add_argument("--character_threshold", type=float, default=1, help="Confidence threshold for adding character category labels")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for images in subfolders")
    parser.add_argument("--remove_underscore", action="store_true", help="Replace underscores with spaces in the output labels")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--undesired_tags", type=str, default="", help="Comma-separated list of tags to remove from the output")
    parser.add_argument('--frequency_tags', action='store_true', help='Display the frequency of image tags')

    args = parser.parse_args()

    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention

    main(args)

