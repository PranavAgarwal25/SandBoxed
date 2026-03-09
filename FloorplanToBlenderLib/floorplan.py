import json

from . import const
from . import config




def new_floorplan(config):

    return floorplan(config)


class floorplan:


    def __init__(self, conf=None):
        if conf is None:

            conf = const.IMAGE_DEFAULT_CONFIG_FILE_NAME
        self.conf = conf
        self.create_variables_from_config(self.conf)

    def __str__(self):
        return str(vars(self))

    def create_variables_from_config(self, conf):
        settings = config.get_all(conf)
        settings_dict = {s: dict(settings.items(s)) for s in settings.sections()}
        for group in settings_dict.items():
            for item in group[1].items():
                setattr(self, item[0], json.loads(item[1]))



