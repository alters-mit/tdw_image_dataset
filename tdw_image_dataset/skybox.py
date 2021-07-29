class Skybox:
    """
    Metadata for setting the HDRI skybox.
    """

    def __init__(self, hdri_index: int, skybox_count: int, command: dict):
        """
        :param hdri_index: The index of the HDRI Skybox record.
        :param skybox_count: The current count of skyboxes.
        :param command: The command to set the skybox.
        """

        """:field
        The index of the HDRI Skybox record.
        """
        self.hdri_index: int = hdri_index
        """:field
        The current count of skyboxes.
        """
        self.skybox_count = skybox_count
        """:field
        The command to set the skybox.
        """
        self.command: dict = command
