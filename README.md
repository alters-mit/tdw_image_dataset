# TDW Image Dataset

Generate a dataset of synthetic images using [TDW](https://github.com/threedworld-mit/tdw). By default, datasets have 1300000 "train" images and 50000 "val" images. A full dataset requires approximately 8 hours to generate using high-end hardware.

## Requirements

[See TDW's requirements](https://github.com/threedworld-mit/tdw/blob/master/Documentation/getting_started.md#requirements)

## Install

1. `git clone https://github.com/alters-mit/tdw_image_dataset`
2. `cd tdw_image_dataset`
3. `pip3 install -e .`
4. `python3 download_build.py` (This will download the correct version of the build)

## Upgrade

1. `cd tdw_image_dataset`
2. `git pull`
3. `python3 download_build.py`

## Changelog

[Read this.](doc/changelog.md)

## Usage

See [`single_env.py`](https://github.com/alters-mit/tdw_image_dataset/blob/main/controllers/single_env.py), [`multi_env.py`](https://github.com/alters-mit/tdw_image_dataset/blob/main/controllers/multi_env.py), and [`test.py`](https://github.com/alters-mit/tdw_image_dataset/blob/main/controllers/test.py). You may want to run `test.py` first to make sure that you can generate image datasets.

## How It Works

The [`ImageDataset`](doc/image_dataset.md) class will search the records databases for all model categories in the TDW library. It will then add the object to the scene, and then generate a target number of images per category, using all models in that category. Each model is added in sequentially to the scene; there is always exactly 1 model in the scene.

To increase variability, each image has randomized camera and positional parameters, and may have additional random parameters, such as the angle of sunlight or the visual materials of the model. This randomness is constrained somewhat in order to guarantee a degree of compositional quality (namely, that the object is guaranteed to always be at least partially in the frame).

### 1. Generate metadata

Every dataset has an associated `metadata.txt` file, which contains a serialized JSON object of all of the parameters used to initialize this dataset. This can be very useful if you are generating many datasets with slightly different parameters.

### 2. Initialize the scene

Each dataset uses exactly 1 scene (`multi_env.py` sidesteps this limitation by running 6 datasets sequentially). The scene's global parameters and post-processing parameters are initialized.

Each scene has one more more "environments", which are spatial boxes in which you expect images to look reasonable. It is possible in TDW to instantiate objects and avatars beyond these limits, but they will be in a blank void. In `ImageDataset`, the avatar and object positions are always constrained to the scene's environments; in interior scenes these are rooms and in exterior scenes there is usually only one environment.`initialize_scene` returns a list of these environments.

### 3. Fetch records

The controller fetches a list of all model categories ("wnids") in the model library.

### 4. Iterate through each wnid

The controller fetches a list of all records in the wnid. If the model has been "processed" (that is if all images for this model have already been generated), the model is skipped over.

### 5. Iterate through each model

#### 5a. Set the starting index

Images are always saved as `<filename>_<index>.jpg`. If the `no_overwrite` in the constructor is set to `False`, the starting index is always `0000`. Otherwise, the starting index will be the number after the last index (if any). This is mostly useful for cases like `multi_env.py` in which you don't want sequential datasets to overwrite each other's images. If you're using only one scene, you probably want images to be overwritten to avoid generating extras if you have to restart the controller.

#### 5b. Add the object and set the scene for grayscale capture

_If you see the window become tiny, this is expected behavior!_

To generate images, `ImageDataset` runs each model through two loops. The first loop captures camera and object positions, rotations, etc. Then, these cached positions are played back in the second loop to generate images. Image capture is divided this way because the first loop will "reject" a lot of images with poor composition; this rejection system doesn't require image data, and so sending image data would slow down the entire controller.

Each object, once instantiated, is set to "unit scale", with its longest extent being set to 1 meter. This way, `ImageDataset` can reliably frame every object using the same positional and rotational parameters.

#### 5c. Positional Loop

Gather valid [`ImagePosition`](doc/image_position.md) data until the list of `ImagePosition` objects equals the target number of images to capture.

`ImageDataset` relies on [`Occlusion`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/api/output_data.md#Occlusion) data to determine whether an image has good composition. This data reduces the rendered frame of two `_mask` passes. One `_mask` pass includes environment objects (such as walls and trees) and the other doesn't. Both images are reduced to a single pixel. The returned value is the difference between the grayscale values of each pixel. This isn't much information but it's enough for this use-case and it's very fast. It also doesn't need a large window size to be useful; in fact, it runs faster if the window is smaller. So, to start the positional loop, the entire window is resized to 32x32 and render quality is set to minimal.

If an object is too occluded, the `ImagePosition` is rejected.

#### 5d. Image Loop

Once `ImagePosition` has enough cached `ImagePosition` data, it can begin to actually generate images. Image quality is now set to maximum, and the screen size is set to the desired image capture size (by default, 256x256).

Every iteration, the object and avatar are positioned and rotated according to the cached `ImagePosition` data. Image data is received and written to disk. This image saving is handled via threading to prevent the controller from slowing down.

##### Optional Additional Commands

- If the `materials` parameter of the constructor is set to `True`: Per frame, all of the object's visual materials will be randomly set to materials from the material library.
- If the `hdri` parameter of the constructor is set to `True`: Periodically set a new HDRI skybox. Per frame, set a random rotation for the HDRI skybox.

#### 5e. Cleanup

Destroy the model and unload its asset bundle from memory.

### 6. Create a .zip file

After generating the whole dataset, `ImageDataset` will zip the dataset directory and destroy the original files. If you don't want the controller to do this, set `do_zip` to `False` in the constructor.

## Known Limitations

- `ImageDataset` can't include physics simulations. If it allowed objects to "fall" or otherwise move before image capture, the positional loop wouldn't work at all (because the object would immediately fall out of frame during the "optimal" pass). For physics datasets, see [tdw_physics](https://github.com/alters-mit/tdw_physics).
- `ImageDataset` only works if there's one model in the scene. All of the image composition logic assumes that there is only one object to rotate, position, frame, etc. Additionally, images are semantically tagged assuming that there's only one object in the image.