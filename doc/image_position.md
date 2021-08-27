# ImagePosition

`from tdw_image_dataset.image_position import ImagePosition`

The positions and rotations of the avatar and object for an image.

Positions are stored as (x, y, z) dictionaries, for example: `{"x": 0, "y": 0, "z": 0}`.
Rotations are stored as (x, y, z, w) dictionaries, for example: `{"x": 0, "y": 0, "z": 0, "w": 1}`.

***

## Fields

- `avatar_position` The position of the avatar.

- `camera_rotation` The rotation of the avatar.

- `object_position` The position of the object.

- `object_rotation` The rotation of the object.

***

## Functions

#### \_\_init\_\_

**`ImagePosition(avatar_position, camera_rotation, object_position, object_rotation)`**

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| avatar_position |  Dict[str, float] |  | The position of the avatar. |
| camera_rotation |  Dict[str, float] |  | The rotation of the avatar. |
| object_position |  Dict[str, float] |  | The position of the object. |
| object_rotation |  Dict[str, float] |  | The rotation of the object. |

