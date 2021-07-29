import os
from secrets import token_urlsafe
from pathlib import Path
import json
from datetime import datetime
from threading import Thread
from time import time
from typing import List, Dict, Tuple, Optional
from zipfile import ZipFile
from distutils import dir_util
import numpy as np
from tqdm import tqdm
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.output_data import Occlusion, Images, ImageSensors, Transforms
from tdw.librarian import ModelLibrarian, MaterialLibrarian, HDRISkyboxLibrarian, ModelRecord, HDRISkyboxRecord
from tdw.scene.scene_bounds import SceneBounds
from tdw.scene.room_bounds import RoomBounds
from tdw_image_dataset.image_position import ImagePosition
from tdw_image_dataset.skybox import Skybox


RNG: np.random.RandomState = np.random.RandomState(0)


class ImageDataset(Controller):
    def __init__(self,
                 port: int = 1071,
                 launch_build: bool = False,
                 materials: bool = False,
                 new: bool = False,
                 screen_width: int = 256,
                 screen_height: int = 256,
                 output_scale: float = 1,
                 hdri: bool = True,
                 show_objects: bool = True,
                 clamp_rotation: bool = True,
                 max_height: float = 0.5,
                 occlusion: float = 0.45,
                 less_dark: bool = True,
                 id_pass: bool = False,
                 overwrite: bool = True,
                 do_zip: bool = True,
                 train: int = 1300000,
                 val: int = 50000,
                 library: str = "models_core.json",
                 random_seed: int = 0):
        """
        :param port: The port used to connect to the build.
        :param launch_build: If True, automatically launch the build. Always set this to False on a Linux server.
        :param materials: If True, set random visual materials for each sub-mesh of each object.
        :param new: If True, clear the list of models that have already been used.
        :param screen_width: The screen width of the build in pixels.
        :param screen_height: The screen height of the build in pixels.
        :param output_scale: Scale the images by this factor before saving to disk.
        :param hdri: If True, use a random HDRI skybox per frame.
        :param show_objects: If True, show objects.
        :param clamp_rotation: If true, clamp the rotation to +/- 30 degrees around each axis.
        :param max_height: The percentage of the environment height that is the ceiling for the avatar and object. Must be between 0 and 1.
        :param occlusion: The occlusion threshold. Lower value = slower FPS, better composition. Must be between 0 and 1.
        :param less_dark: If True, there will be more daylight exterior skyboxes (requires hdri == True)
        :param id_pass: If True, send and save the _id pass.
        :param overwrite: If True, overwrite existing images.
        :param do_zip: If True, zip the directory at the end.
        :param train: The number of train images.
        :param val: The number of val images.
        :param library: The path to the library records file.
        :param random_seed: The random seed.
        """

        global RNG
        RNG = np.random.RandomState(random_seed)

        """:field
        The width of the build screen in pixels.
        """
        self.screen_width: int = screen_width
        """:field
        The height of the screen in pixels.
        """
        self.screen_height: int = screen_height
        """:field
        Scale all images to this size in pixels before writing to disk.
        """
        self.output_size: Tuple[int, int] = (int(screen_width * output_scale), int(screen_height * output_scale))
        """:field
        If True, scale images before writing to disk.
        """
        self.scale: bool = output_scale != 1
        """:field
        If True, show objects.
        """
        self.show_objects: bool = show_objects
        """:field
        If true, clamp the rotation to +/- 30 degrees around each axis.
        """
        self.clamp_rotation: bool = clamp_rotation
        """:field
        The percentage of the environment height that is the ceiling for the avatar and object. Must be between 0 and 1.
        """
        self.max_height: float = max_height
        """:field
        The occlusion threshold. Lower value = slower FPS, better composition. Must be between 0 and 1.
        """
        self.occlusion: float = occlusion
        """:field
        If True, send and save the _id pass.
        """
        self.id_pass: bool = id_pass
        """:field
        If True, overwrite existing images.
        """
        self.overwrite: bool = overwrite
        """:field
        If True, zip the directory at the end.
        """
        self.do_zip: bool = do_zip
        """:field
        The number of train images.
        """
        self.train: int = train
        """:field
        The number of val images.
        """
        self.val: int = val

        assert 0 < max_height <= 1.0, f"Invalid max height: {max_height}"
        assert 0 < occlusion <= 1.0, f"Invalid occlusion threshold: {occlusion}"

        """:field
        If True, there will be more daylight exterior skyboxes (requires hdri == True)
        """
        self.less_dark: bool = less_dark
        """:field
        Cached model substructure data.
        """
        self.substructures: Dict[str, List[dict]] = dict()
        """:field
        Cached initial (canonical) rotations per model.
        """
        self.initial_rotations: Dict[str, Dict[str, float]] = dict()
        """:field
        If True, clear the list of models that have already been used.
        """
        self.new: bool = new
        """:field
        If True, set random visual materials for each sub-mesh of each object.
        """
        self.materials: bool = materials

        super().__init__(port=port, launch_build=launch_build)

        self.model_librarian = ModelLibrarian(library=library)
        self.material_librarian = MaterialLibrarian("materials_low.json")
        self.hdri_skybox_librarian = HDRISkyboxLibrarian()
        """:field
        Cached skybox records.
        """
        self.skyboxes: Optional[List[HDRISkyboxRecord]] = None
        # Get skybox records.
        if hdri:
            self.skyboxes: List[HDRISkyboxRecord] = self.hdri_skybox_librarian.records
            # Prefer exterior daytime skyboxes by adding them multiple times to the list.
            if self.less_dark:
                temp = self.skyboxes[:]
                for skybox in temp:
                    if skybox.location != "interior" and skybox.sun_elevation >= 145:
                        self.skyboxes.append(skybox)

    def initialize_scene(self, scene_command, a="a") -> SceneBounds:
        """
        Initialize the scene.

        :param scene_command: The command to load the scene.
        :param a: The avatar ID.

        :return: The [`SceneBounds`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/scene_bounds.md) of the scene.
        """

        # Initialize the scene.
        # Add the avatar.
        commands = [scene_command,
                    {"$type": "create_avatar",
                     "type": "A_Img_Caps_Kinematic",
                     "id": a}]
        # Disable physics.
        # Enable jpgs.
        # Set FOV.
        # Set clipping planes.
        # Set AA.
        # Set aperture.
        # Disable vignette.
        commands.extend([{"$type": "simulate_physics",
                          "value": False},
                         {"$type": "set_img_pass_encoding",
                          "value": False},
                         {'$type': 'set_field_of_view',
                          'avatar_id': a,
                          'field_of_view': 60},
                         {'$type': 'set_camera_clipping_planes',
                          'avatar_id': a,
                          'far': 160,
                          'near': 0.01},
                         {"$type": "set_anti_aliasing",
                          "avatar_id": a,
                          "mode": "subpixel"},
                         {"$type": "set_aperture",
                          "aperture": 70},
                         {'$type': 'set_vignette',
                          'enabled': False}])

        # If we're using HDRI skyboxes, send additional favorable post-process commands.
        if self.skyboxes is not None:
            commands.extend([{"$type": "set_post_exposure",
                              "post_exposure": 0.6},
                             {"$type": "set_contrast",
                              "contrast": -20},
                             {"$type": "set_saturation",
                              "saturation": 10},
                             {"$type": "set_screen_space_reflections",
                              "enabled": False},
                             {"$type": "set_shadow_strength",
                              "strength": 1.0},
                             {"$type": "send_environments"}])
        # Send the commands.
        resp = self.communicate(commands)
        return SceneBounds(resp[0])

    def generate_metadata(self, dataset_dir: str, scene_name: str) -> None:
        """
        Generate a metadata file for this dataset.

        :param dataset_dir: The dataset directory for images.
        :param scene_name: The scene name.
        """

        root_dir = f"{dataset_dir}/images/"
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        data = {"dataset": dataset_dir,
                "scene": scene_name,
                "train": self.train,
                "val": self.val,
                "materials": self.materials is not None,
                "hdri": self.skyboxes is not None,
                "screen_width": self.screen_width,
                "screen_height": self.screen_height,
                "output_scale": self.scale,
                "clamp_rotation": self.clamp_rotation,
                "show_objects": self.show_objects,
                "max_height": self.max_height,
                "occlusion": self.occlusion,
                "less_dark": self.less_dark,
                "start": datetime.now().strftime("%H:%M %d.%m.%y")}
        with open(os.path.join(root_dir, "metadata.txt"), "wt") as f:
            json.dump(data, f, sort_keys=True, indent=4)

    def run(self, dataset_dir: str, scene_name: str) -> None:
        """
        Generate the dataset.

        :param dataset_dir: The dataset directory for images.
        :param scene_name: The scene name.
        """

        # Create the metadata file.
        self.generate_metadata(dataset_dir,
                               scene_name=scene_name)

        # The root directory of the output.
        root_dir = f"{dataset_dir}/images/"

        # The avatar ID.
        a = "a"

        # Initialize the scene.
        scene_bounds: SceneBounds = self.initialize_scene(self.get_add_scene(scene_name))

        # Fetch the WordNet IDs.
        wnids = self.model_librarian.get_model_wnids()
        # Remove any wnids that don't have valid models.
        wnids = [w for w in wnids if len(
            [r for r in self.model_librarian.get_all_models_in_wnid(w) if not r.do_not_use]) > 0]

        # Set the number of train and val images per wnid.
        num_train = self.train / len(wnids)
        num_val = self.val / len(wnids)

        # Create the progress bar.
        pbar = tqdm(total=len(wnids))

        # If this is a new dataset, remove the previous list of completed models.
        done_models_path: Path = Path(dataset_dir).joinpath("processed_records.txt")
        if self.new and done_models_path.exists():
            done_models_path.unlink()

        # Get a list of models that have already been processed.
        processed_model_names: List[str] = []
        if done_models_path.exists():
            processed_model_names = done_models_path.read_text(encoding="utf-8").split("\n")

        # Iterate through each wnid.
        for w, q in zip(wnids, range(len(wnids))):
            # Update the progress bar.
            pbar.set_description(w)

            # Get all valid models in the wnid.
            records = self.model_librarian.get_all_models_in_wnid(w)
            records = [r for r in records if not r.do_not_use]

            # Get the train and val counts.
            train_count = [len(a) for a in np.array_split(
                np.arange(num_train), len(records))][0]
            val_count = [len(a) for a in np.array_split(
                np.arange(num_val), len(records))][0]

            # Process each record.
            fps = "nan"
            for record, i in zip(records, range(len(records))):
                # Set the progress bar description to the wnid and FPS.
                pbar.set_description(f"record {i + 1}/{len(records)}, FPS {fps}")

                # Skip models that have already been processed.
                if record.name in processed_model_names:
                    continue

                # Create all of the images for this model.
                dt = self.process_model(record, a, scene_bounds, train_count, val_count, root_dir, w)
                fps = round((train_count + val_count) / dt)

                # Mark this record as processed.
                with done_models_path.open("at") as f:
                    f.write(f"\n{record.name}")
            pbar.update(1)
        pbar.close()

        # Add the end time to the metadata file.
        with open(os.path.join(root_dir, "metadata.txt"), "rt") as f:
            data = json.load(f)
            end_time = datetime.now().strftime("%H:%M %d.%m.%y")
            if "end" in data:
                data["end"] = end_time
            else:
                data.update({"end": end_time})
        with open(os.path.join(root_dir, "metadata.txt"), "wt") as f:
            json.dump(data, f, sort_keys=True, indent=4)

        # Terminate the build.
        if self.overwrite:
            self.communicate({"$type": "terminate"})
        # Zip up the images.
        if self.do_zip:
            zip_dir = Path(dataset_dir)
            ImageDataset.zip_images(zip_dir)

    def set_skybox(self, records: List[HDRISkyboxRecord], its_per_skybox: int, hdri_index: int, skybox_count: int) -> Skybox:
        """
        If it's time, set a new skybox.

        :param records: All HDRI records.
        :param its_per_skybox: Iterations per skybox.
        :param hdri_index: The index in the records list.
        :param skybox_count: The number of images of this model with this skybox.

        :return: Data for setting the skybox.
        """

        # Set a new skybox.
        if skybox_count == 0:
            command = self.get_add_hdri_skybox(records[hdri_index].name)
        # It's not time yet to set a new skybox. Don't send a command.
        else:
            command = None
        skybox_count += 1
        if skybox_count >= its_per_skybox:
            hdri_index += 1
            if hdri_index >= len(records):
                hdri_index = 0
            skybox_count = 0
        return Skybox(hdri_index, skybox_count, command)

    def process_model(self, record: ModelRecord, a: str, scene_bounds: SceneBounds, train_count: int, val_count: int,
                      root_dir: str, wnid: str) -> float:
        """
        Capture images of a model.

        :param record: The model record.
        :param a: The ID of the avatar.
        :param scene_bounds: The bounds of the scene.
        :param train_count: Number of train images.
        :param val_count: Number of val images.
        :param root_dir: The root directory for saving images.
        :param wnid: The wnid of the record.
        :return The time elapsed.
        """

        image_count = 0

        # Get the filename index. If we shouldn't overwrite any images, start after the last image.
        if not self.overwrite:
            # Check if any images exist.
            wnid_dir = Path(root_dir).joinpath(f"train/{wnid}")
            if wnid_dir.exists():
                max_file_index = -1
                for image in wnid_dir.iterdir():
                    if not image.is_file() or image.suffix != ".jpg" \
                            or not image.stem.startswith("img_") or image.stem[4:-5] != record.name:
                        continue
                    image_index = int(image.stem[-4:])
                    if image_index > max_file_index:
                        max_file_index = image_index
                file_index = max_file_index + 1
            else:
                file_index = 0
        else:
            file_index = 0

        image_positions: List[ImagePosition] = []
        o_id = self.get_unique_id()

        s = TDWUtils.get_unit_scale(record)

        # Add the object.
        # Set the screen size to 32x32 (to make the build run faster; we only need the average grayscale values).
        # Toggle off pass masks.
        # Set render quality to minimal.
        # Scale the object to "unit size".
        resp = self.communicate([{"$type": "add_object",
                                  "name": record.name,
                                  "url": record.get_url(),
                                  "scale_factor": record.scale_factor,
                                  "category": record.wcategory,
                                  "rotation": record.canonical_rotation,
                                  "id": o_id},
                                 {"$type": "set_screen_size",
                                  "height": 32,
                                  "width": 32},
                                 {"$type": "set_pass_masks",
                                  "avatar_id": a,
                                  "pass_masks": []},
                                 {"$type": "set_render_quality",
                                  "render_quality": 0},
                                 {"$type": "scale_object",
                                  "id": o_id,
                                  "scale_factor": {"x": s, "y": s, "z": s}},
                                 {"$type": "send_transforms"}])
        # Cache the initial rotation of the object.
        if record.name not in self.initial_rotations:
            self.initial_rotations[record.name] = TDWUtils.array_to_vector4(Transforms(resp[0]).get_rotation(0))
        # The index in the HDRI records array.
        hdri_index = 0
        # The number of iterations on this skybox so far.
        skybox_count = 0
        if self.skyboxes:
            # The number of iterations per skybox for this model.
            its_per_skybox = round((train_count + val_count) / len(self.skyboxes))

            # Set the first skybox.
            skybox: Skybox = self.set_skybox(self.skyboxes, its_per_skybox, hdri_index, skybox_count)
            hdri_index = skybox.hdri_index
            skybox_count = skybox.skybox_count
            self.communicate(skybox.command)
        else:
            its_per_skybox = 0

        while len(image_positions) < train_count + val_count:
            # Get a random "room".
            room: RoomBounds = scene_bounds.rooms[RNG.randint(0, len(scene_bounds.rooms))]
            # Get the occlusion
            occlusion, image_position = self.get_occlusion(record.name, o_id, a, room)
            if occlusion < self.occlusion:
                image_positions.append(image_position)
        # Send images.
        # Set the screen size.
        # Set render quality to maximum.
        commands = [{"$type": "send_images",
                     "frequency": "always"},
                    {"$type": "set_pass_masks",
                     "avatar_id": a,
                     "pass_masks": ["_img", "_id"] if self.id_pass else ["_img"]},
                    {"$type": "set_screen_size",
                     "height": self.screen_height,
                     "width": self.screen_width},
                    {"$type": "set_render_quality",
                     "render_quality": 5}]
        # Hide the object maybe.
        if not self.show_objects:
            commands.append({"$type": "hide_object",
                             "id": o_id})
        self.communicate(commands)

        # Generate images from the cached spatial data.
        t0 = time()
        train = 0
        for p in image_positions:
            # Teleport the avatar.
            # Rotate the avatar's camera.
            # Teleport the object.
            # Rotate the object.
            # Get the response.
            commands = [{"$type": "teleport_avatar_to",
                         "avatar_id": a,
                         "position": p.avatar_position},
                        {"$type": "rotate_sensor_container_to",
                         "avatar_id": a,
                         "rotation": p.camera_rotation},
                        {"$type": "teleport_object",
                         "id": o_id,
                         "position": p.object_position},
                        {"$type": "rotate_object_to",
                         "id": o_id,
                         "rotation": p.object_rotation}]
            # Set the visual materials.
            if self.materials is not None:
                if record.name not in self.substructures:
                    self.substructures[record.name] = record.substructure
                for sub_object in self.substructures[record.name]:
                    for i in range(len(sub_object["materials"])):
                        material_name = self.material_librarian.records[RNG.randint(0, len(self.material_librarian.records))].name
                        commands.extend([self.get_add_material(material_name),
                                         {"$type": "set_visual_material",
                                          "id": o_id,
                                          "material_name": material_name,
                                          "object_name": sub_object["name"],
                                          "material_index": i}])
            # Maybe set a new skybox.
            # Rotate the skybox.
            if self.skyboxes:
                hdri_index, skybox_count, command = self.set_skybox(self.skyboxes, its_per_skybox, hdri_index,
                                                                    skybox_count)
                if command:
                    commands.append(command)
                commands.append({"$type": "rotate_hdri_skybox_by",
                                 "angle": RNG.uniform(0, 360)})

            resp = self.communicate(commands)

            # Create a thread to save the image.
            t = Thread(target=self.save_image, args=(resp, record, file_index, root_dir, wnid, train, train_count))
            t.daemon = True
            t.start()
            train += 1
            file_index += 1
            image_count += 1
        t1 = time()

        # Stop sending images.
        # Destroy the object.
        # Unload asset bundles.
        self.communicate([{"$type": "send_images",
                           "frequency": "never"},
                          {"$type": "destroy_object",
                           "id": o_id},
                          {"$type": "unload_asset_bundles"}])
        return t1 - t0

    def save_image(self, resp, record: ModelRecord, image_count: int, root_dir: str, wnid: str, train: int,
                   train_count: int) -> None:
        """
        Save an image.

        :param resp: The raw response data.
        :param record: The model record.
        :param image_count: The image count.
        :param root_dir: The root directory.
        :param wnid: The wnid.
        :param train: Number of train images so far.
        :param train_count: Total number of train images to generate.
        """

        # Get the directory.
        directory: Path = Path(root_dir).joinpath("train" if train < train_count else "val").joinpath(wnid)
        if directory.exists():
            # Try to make the directories. Due to threading, they might already be made.
            try:
                directory.mkdir(parents=True)
            except OSError:
                pass

        # Save the image.
        filename = f"{record.name}_{image_count:04d}"

        # Save the image without resizing.
        if not self.scale:
            TDWUtils.save_images(Images(resp[0]), filename,
                                 output_directory=directory)
        # Resize the image and save it.
        else:
            TDWUtils.save_images(Images(resp[0]), filename,
                                 output_directory=directory,
                                 resize_to=self.output_size)

    def get_occlusion(self, o_name: str, o_id: int, a_id: str, room: RoomBounds) -> Tuple[float, ImagePosition]:
        """
        Get the "real" grayscale value of an image we hope to capture.

        :param o_name: The name of the object.
        :param o_id: The ID of the object.
        :param a_id: The ID of the avatar.
        :param room: The "room" bounds.

        :return: (grayscale, distance, avatar_position, object_position, object_rotation, avatar_rotation)
        """

        # Get a random position for the avatar.
        a_p = np.array([RNG.uniform(room.x_min, room.x_max),
                        RNG.uniform(0.4, room.y_max),
                        RNG.uniform(room.z_min, room.z_max)])

        # Get a random distance from the avatar.
        d = RNG.uniform(0.8, 3)

        # Get a random position for the object constrained to the environment bounds.
        o_p = ImageDataset.sample_spherical() * d
        # Clamp the y value to positive.
        o_p[1] = abs(o_p[1])
        o_p = a_p + o_p

        # Clamp the y value of the object.
        if o_p[1] > room.y_max:
            o_p[1] = room.y_max

        # Convert the avatar's position to a Vector3.
        a_p = TDWUtils.array_to_vector3(a_p)

        # Set random camera rotations.
        yaw = RNG.uniform(-15, 15)
        pitch = RNG.uniform(-15, 15)

        # Convert the object position to a Vector3.
        o_p = TDWUtils.array_to_vector3(o_p)

        # Add rotation commands.
        # If we're clamping the rotation, rotate the object within +/- 30 degrees on each axis.
        if self.clamp_rotation:
            o_rot = None
            commands = [{"$type": "rotate_object_to",
                         "id": o_id,
                         "rotation": self.initial_rotations[o_name]},
                        {"$type": "rotate_object_by",
                         "id": o_id,
                         "angle": RNG.uniform(-30, 30),
                         "axis": "pitch"},
                        {"$type": "rotate_object_by",
                         "id": o_id,
                         "angle": RNG.uniform(-30, 30),
                         "axis": "yaw"},
                        {"$type": "rotate_object_by",
                         "id": o_id,
                         "angle": RNG.uniform(-30, 30),
                         "axis": "roll"}]
        # Set a totally random rotation.
        else:
            o_rot = {"x": RNG.uniform(-360, 360),
                     "y": RNG.uniform(-360, 360),
                     "z": RNG.uniform(-360, 360),
                     "w": RNG.uniform(-360, 360)}
            commands = [{"$type": "rotate_object_to",
                         "id": o_id,
                         "rotation": o_rot}]

        # After rotating the object:
        # 1. Teleport the object.
        # 2. Teleport the avatar.
        # 3. Look at the object.
        # 4. Perturb the camera slightly.
        # 5. Send grayscale data and image sensor data.
        commands.extend([{"$type": "teleport_object",
                          "id": o_id,
                          "position": o_p},
                         {"$type": "teleport_avatar_to",
                          "avatar_id": a_id,
                          "position": a_p},
                         {"$type": "look_at",
                          "avatar_id": a_id,
                          "object_id": o_id,
                          "use_centroid": True},
                         {"$type": "rotate_sensor_container_by",
                          "angle": pitch,
                          "avatar_id": a_id,
                          "axis": "pitch"},
                         {"$type": "rotate_sensor_container_by",
                          "angle": yaw,
                          "avatar_id": a_id,
                          "axis": "yaw"},
                         {"$type": "send_occlusion",
                          "frequency": "once"},
                         {"$type": "send_image_sensors",
                          "frequency": "once"}])
        # If we clamped the rotation of the object, we need to know its quaternion.
        if self.clamp_rotation:
            commands.append({"$type": "send_transforms",
                             "frequency": "once",
                             "ids": [o_id]})

        # Send the commands.
        resp = self.communicate(commands)

        # Parse the output data:
        # 1. The occlusion value of the image.
        # 2. The camera rotation.
        occlusion: float = 0
        cam_rot = None
        for i in range(len(resp) - 1):
            r_id = resp[i][4:8]
            if r_id == b"occl":
                occlusion = Occlusion(resp[i]).get_occluded()
            elif r_id == b"imse":
                cam_rot = ImageSensors(resp[i]).get_sensor_rotation(0)
                cam_rot = {"x": cam_rot[0], "y": cam_rot[1], "z": cam_rot[2], "w": cam_rot[3]}
            elif r_id == b"tran":
                o_rot = TDWUtils.array_to_vector4(Transforms(resp[i]).get_rotation(0))
        return occlusion, ImagePosition(avatar_position=a_p,
                                        object_position=o_p,
                                        object_rotation=o_rot,
                                        camera_rotation=cam_rot)

    @staticmethod
    def sample_spherical(npoints=1, ndim=3) -> np.array:
        vec = RNG.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return np.array([vec[0][0], vec[1][0], vec[2][0]])

    @staticmethod
    def zip_images(zip_dir: Path) -> None:
        """
        Zip up the images.

        :param zip_dir: The zip directory.
        """

        if not zip_dir.exists():
            zip_dir.mkdir()

        # Use a random token to avoid overwriting zip files.
        token = token_urlsafe(4)
        zip_path = zip_dir.joinpath(f"images_{token}.zip")
        images_directory = str(zip_dir.joinpath("images").resolve())

        # Source: https://thispointer.com/python-how-to-create-a-zip-archive-from-multiple-files-or-directory/
        with ZipFile(str(zip_path.resolve()), 'w') as zip_obj:
            # Iterate over all the files in directory
            for folderName, subfolders, filenames in os.walk(images_directory):
                for filename in filenames:
                    # create complete filepath of file in directory
                    file_path = os.path.join(folderName, filename)
                    # Add file to zip
                    zip_obj.write(file_path, os.path.basename(file_path))
        # Remove the original images.
        dir_util.remove_tree(images_directory)
