# ImageDataset

`from tdw_image_dataset.image_dataset import ImageDataset`

Generate image datasets. Each image will have a single object in the scene in a random position and orientation.
Optionally, the scene might have variable lighting and the object might have variable visual materials.

The image dataset includes all models in a model library (the default library is models_core.json) sorted by wnid and model name.

***

## Class Variables

| Variable | Type | Description |
| --- | --- | --- |
| `AVATAR_ID` | str | The ID of the avatar. |

***

## Fields

- `output_directory` The root output directory.

- `images_directory` The images output directory.

- `metadata_path` The path to the metadata file.

- `screen_width` The width of the build screen in pixels.

- `screen_height` The height of the screen in pixels.

- `output_size` Scale all images to this size in pixels before writing to disk.

- `scale` If True, scale images before writing to disk.

- `show_objects` If True, show objects.

- `clamp_rotation` If true, clamp the rotation to +/- 30 degrees around each axis.

- `max_height` The percentage of the environment height that is the ceiling for the avatar and object. Must be between 0 and 1.

- `occlusion` The occlusion threshold. Lower value = slower FPS, better composition. Must be between 0 and 1.

- `id_pass` If True, send and save the _id pass.

- `overwrite` If True, overwrite existing images.

- `do_zip` If True, zip the directory at the end.

- `train` The number of train images.

- `val` The number of val images.

- `less_dark` If True, there will be more daylight exterior skyboxes (requires hdri == True)

- `substructures` Cached model substructure data.

- `initial_rotations` Cached initial (canonical) rotations per model.

- `new` If True, clear the list of models that have already been used.

- `materials` If True, set random visual materials for each sub-mesh of each object.

- `skyboxes` Cached skybox records.

***

## Functions

#### \_\_init\_\_

**`ImageDataset(output_directory)`**

**`ImageDataset(output_directory, port=1071, launch_build=False, materials=False, new=False, screen_width=256, screen_height=256, output_scale=1, hdri=True, show_objects=True, clamp_rotation=True, max_height=0.5, occlusion=0.45, less_dark=True, id_pass=False, overwrite=True, do_zip=True, train=1300000, val=50000, library="models_core.json", random_seed=0)`**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| output_directory |  Union[str, Path] |  | The path to the root output directory. |
| port |  int  | 1071 | The port used to connect to the build. |
| launch_build |  bool  | False | If True, automatically launch the build. Always set this to False on a Linux server. |
| materials |  bool  | False | If True, set random visual materials for each sub-mesh of each object. |
| new |  bool  | False | If True, clear the list of models that have already been used. |
| screen_width |  int  | 256 | The screen width of the build in pixels. |
| screen_height |  int  | 256 | The screen height of the build in pixels. |
| output_scale |  float  | 1 | Scale the images by this factor before saving to disk. |
| hdri |  bool  | True | If True, use a random HDRI skybox per frame. |
| show_objects |  bool  | True | If True, show objects. |
| clamp_rotation |  bool  | True | If true, clamp the rotation to +/- 30 degrees around each axis. |
| max_height |  float  | 0.5 | The percentage of the environment height that is the ceiling for the avatar and object. Must be between 0 and 1. |
| occlusion |  float  | 0.45 | The occlusion threshold. Lower value = slower FPS, better composition. Must be between 0 and 1. |
| less_dark |  bool  | True | If True, there will be more daylight exterior skyboxes (requires hdri == True) |
| id_pass |  bool  | False | If True, send and save the _id pass. |
| overwrite |  bool  | True | If True, overwrite existing images. |
| do_zip |  bool  | True | If True, zip the directory at the end. |
| train |  int  | 1300000 | The number of train images. |
| val |  int  | 50000 | The number of val images. |
| library |  str  | "models_core.json" | The path to the library records file. |
| random_seed |  int  | 0 | The random seed. |

#### initialize_scene

**`self.initialize_scene(scene_command)`**

Initialize the scene.


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| scene_command |  |  | The command to load the scene. |

_Returns:_  The [`SceneBounds`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/python/scene_bounds.md) of the scene.

#### generate_metadata

**`self.generate_metadata(scene_name)`**

Generate a metadata file for this dataset.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| scene_name |  str |  | The scene name. |

#### run

**`self.run(scene_name)`**

Generate the dataset.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| scene_name |  str |  | The scene name. |

#### process_model

**`self.process_model(record, scene_bounds, train_count, val_count, wnid)`**

Capture images of a model.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| record |  ModelRecord |  | The model record. |
| scene_bounds |  SceneBounds |  | The bounds of the scene. |
| train_count |  int |  | Number of train images. |
| val_count |  int |  | Number of val images. |
| wnid |  str |  | The wnid of the record. |

_Returns:_  The time elapsed.

#### get_object_initialization_commands

**`self.get_object_initialization_commands(record, o_id)`**


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| record |  ModelRecord |  | The model record. |
| o_id |  int |  | The object ID. |

_Returns:_  Commands for creating and initializing the object.

#### save_image

**`self.save_image(resp, record, image_count, wnid, train, train_count)`**

Save an image.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| resp |  |  | The raw response data. |
| record |  ModelRecord |  | The model record. |
| image_count |  int |  | The image count. |
| wnid |  str |  | The wnid. |
| train |  int |  | Number of train images so far. |
| train_count |  int |  | Total number of train images to generate. |

#### get_occlusion

**`self.get_occlusion(o_name, o_id, region)`**

Get the "real" grayscale value of an image we hope to capture.


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| o_name |  str |  | The name of the object. |
| o_id |  int |  | The ID of the object. |
| region |  RegionBounds |  | The scene region bounds. |

_Returns:_  (grayscale, distance, avatar_position, object_position, object_rotation, avatar_rotation)

#### get_avatar_position

**`ImageDataset(Controller).get_avatar_position(region)`**

_This is a static function._


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| region |  RegionBounds |  | The scene region bounds. |

_Returns:_  The position of the avatar for the next image as a numpy array.

#### get_object_position_commands

**`self.get_object_position_commands(o_id, avatar_position, region)`**


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| o_id |  int |  | The object ID. |
| avatar_position |  np.array |  | The position of the avatar. |
| region |  RegionBounds |  | The scene region bounds. |

_Returns:_  The position of the object for the next image as a numpy array.

#### get_object_rotation_commands

**`self.get_object_rotation_commands(o_id, o_name)`**


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| o_id |  int |  | The object ID. |
| o_name |  str |  | The object name. |

_Returns:_  A list of commands to rotate the object.

#### get_camera_rotation_commands

**`self.get_camera_rotation_commands(o_id)`**


| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| o_id |  int |  | The object ID. |

_Returns:_  A list of commands to rotate the camera.

#### sample_spherical

**`ImageDataset(Controller).sample_spherical()`**

_This is a static function._

Zip up the images.

#### zip_images

**`self.zip_images()`**

Zip up the images.

