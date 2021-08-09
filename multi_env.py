from pathlib import Path
from argparse import ArgumentParser
from tdw_image_dataset.image_dataset import ImageDataset

"""
Generate a dataset with multiple environments.
"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--library",
                        type=str,
                        default="models_full.json",
                        help="The filename of the model library.")
    parser.add_argument("--dir", type=str, help="The full path of the output directory.")
    args = parser.parse_args()
    scenes = ["building_site",
              "lava_field",
              "iceland_beach",
              "ruin",
              "dead_grotto",
              "abandoned_factory"]
    train = int(1300000 / len(scenes))
    val = int(50000 / len(scenes))
    c = ImageDataset(new=True,
                     clamp_rotation=True,
                     less_dark=True,
                     hdri=True,
                     overwrite=False,
                     max_height=0.5,
                     occlusion=0.45,
                     train=train,
                     val=val,
                     do_zip=False,
                     library=args.library,
                     output_directory=args.dir)

    # Generate a "partial" dataset per scene.
    for scene, i in zip(scenes, range(len(scenes))):
        print(f"{scene}\t{i + 1}/{len(scenes)}")
        c.run(scene_name=scene)
    # Terminate the build.
    c.communicate({"$type": "terminate"})

    # Zip.
    ImageDataset.zip_images(Path(args.dir))
