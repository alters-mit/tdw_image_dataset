from typing import Dict


class ImagePosition:
    """
    The positions and rotations of the avatar and object for an image.

    Positions are stored as (x, y, z) dictionaries, for example: `{"x": 0, "y": 0, "z": 0}`.
    Rotations are stored as (x, y, z, w) dictionaries, for example: `{"x": 0, "y": 0, "z": 0, "w": 1}`.
    """

    def __init__(self, avatar_position: Dict[str, float],
                 camera_rotation: Dict[str, float],
                 object_position: Dict[str, float],
                 object_rotation: Dict[str, float]):
        """
        :param avatar_position: The position of the avatar.
        :param camera_rotation: The rotation of the avatar.
        :param object_position: The position of the object.
        :param object_rotation: The rotation of the object.
        """

        """:field
        The position of the avatar.
        """
        self.avatar_position: Dict[str, float] = avatar_position
        """:field
        The rotation of the avatar.
        """
        self.camera_rotation: Dict[str, float] = camera_rotation
        """:field
        The position of the object.
        """
        self.object_position: Dict[str, float] = object_position
        """:field
        The rotation of the object.
        """
        self.object_rotation: Dict[str, float] = object_rotation
