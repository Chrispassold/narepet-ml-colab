from model import *


class UnetModel:
    def __init__(self, pretrained_weights=None):
        self.model = unet(pretrained_weights=pretrained_weights)
