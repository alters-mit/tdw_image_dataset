from typing import List
import numpy as np
from tdw.librarian import ModelRecord
from tdw.scene.room_bounds import RoomBounds
from tdw_image_dataset.image_dataset import ImageDataset, RNG


class ParentToAvatar(ImageDataset):
    """
    Parent the object to the avatar. When the avatar teleports, the object will also teleport.
    """

    def get_object_initialization_commands(self, record: ModelRecord, o_id: int) -> List[dict]:
        commands = super().get_object_initialization_commands(record=record, o_id=o_id)
        # Parent the object to the avatar and look at the object.
        commands.extend([{'$type': 'teleport_avatar_to',
                          'position': {"x": 1.57, "y": 1., "z": 3.56}},
                         {"$type": "look_at",
                          "object_id": o_id,
                          "use_centroid": True},
                         {"$type": "parent_object_to_avatar",
                          "id": o_id}])
        return commands

    def get_object_position_commands(self, o_id: int, avatar_position: np.array, room: RoomBounds) -> List[dict]:
        return []

    def get_camera_rotation_commands(self, o_id: int) -> List[dict]:
        return [{"$type": "rotate_sensor_container_to",
                 "rotation": {"x": RNG.uniform(-360, 360),
                              "y": RNG.uniform(-360, 360),
                              "z": RNG.uniform(-360, 360),
                              "w": RNG.uniform(-360, 360)}}]
