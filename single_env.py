from argparse import ArgumentParser
from tdw_image_dataset.image_dataset import ImageDataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scene_name", type=str, default="tdw_room",
                        help="The name of the scene. For a complete list: librarian.fetch_all_scene_records()")
    parser.add_argument("--output_dir", type=str, default="D:/Test",
                        help="The absolute path to the output directory.")
    parser.add_argument("--materials", action="store_true", help="Set random visual materials per frame.")
    parser.add_argument("--new", action="store_true", help="Start a new dataset (erases the log of completed models).")
    parser.add_argument("--screen_size", type=int, default=256, help="The screen size of the build.")
    parser.add_argument("--output_scale", type=float, default=1, help="Images are resized by this factor.")
    parser.add_argument("--hdri", action="store_true", help="Use a random HDRI skybox per frame.")
    parser.add_argument("--hide", action="store_true", help="Hide all objects.")
    parser.add_argument("--clamp_rotation", action="store_true",
                        help="Clamp rotation to +/- 30 degrees on each axis, rather than totally random.")
    parser.add_argument("--port", type=int, default=1071, help="The port for the controller and build.")
    parser.add_argument("--launch_build", action="store_true",
                        help="Automatically launch the build. "
                             "Don't add this if you're running the script on a Linux server.")
    parser.add_argument("--max_height", type=float, default=0.5,
                        help="Objects and avatars can be at this percentage of the scene bounds height. Must be between 0 and 1.")
    parser.add_argument("--occlusion", type=float, default=0.45,
                        help="Target occlusion value. Must be between 0 and 1. Lower value = better composition and slower FPS.")
    parser.add_argument("--less_dark", action="store_true", help='Prefer fewer "dark" skyboxes.')
    parser.add_argument("--id_pass", action="store_true", help="Include the _id pass.")
    parser.add_argument("--no_overwrite", action="store_true",
                        help="If true, don't overwrite existing images, and start indexing after the highest index.")
    parser.add_argument("--zip", action="store_true", help="Zip the images after finishing the dataset.")
    parser.add_argument("--train", type=int, default=1300000, help="Total number of train images.")
    parser.add_argument("--val", type=int, default=50000, help="Total number of val images.")
    parser.add_argument("--library", type=str, default="models_core.json",
                        help="The path to the model library records.")
    args = parser.parse_args()

    c = ImageDataset(port=args.port,
                     launch_build=args.launch_build,
                     materials=args.materials,
                     new=args.new,
                     screen_width=args.screen_size,
                     screen_height=args.screen_size,
                     output_scale=args.output_scale,
                     hdri=args.hdri,
                     show_objects=not args.hide,
                     clamp_rotation=args.clamp_rotation,
                     max_height=args.max_height,
                     occlusion=args.occlusion,
                     less_dark=args.less_dark,
                     id_pass=args.id_pass,
                     overwrite=not args.no_overwrite,
                     do_zip=args.zip,
                     train=args.train,
                     val=args.val,
                     library=args.library,
                     output_directory=args.output_director)
    c.run(scene_name=args.scene_name)
