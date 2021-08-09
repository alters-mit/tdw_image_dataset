from pathlib import Path
from tdw_image_dataset.image_dataset import ImageDataset

"""
Test the core functionality of tdw_image_dataset.
"""

if __name__ == "__main__":
    c = ImageDataset(train=20000, val=50, library="models_full.json", materials=True, hdri=False, overwrite=True,
                     do_zip=True, random_seed=0, output_directory=Path.home().joinpath("tdw_image_dataset_test"))
    c.run(scene_name="tdw_room")
