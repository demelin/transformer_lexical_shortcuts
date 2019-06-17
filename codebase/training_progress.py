""" Training progress """

import json


class TrainingProgress(object):
    """ Object used to store, serialize and de-serialize pure python variables that change during training and
    should be preserved in order to properly restart the training process.
    """

    def load_from_json(self, file_name):
        with open(file_name, 'r') as out_file:
            values = json.load(out_file)
        self.__dict__.update(values)

    def save_to_json(self, file_name):
        # Convert strings to bytes
        with open(file_name, 'w') as in_file:
            json.dump(self.__dict__, in_file, indent=2, ensure_ascii=False)
