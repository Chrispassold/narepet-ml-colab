from utils import download_labelbox_json, filter_labelbox_json, split_image_arr


class DataLoader:

    def __init__(self):
        self.raw_data = []

    def load_from_labelbox_api(self, api_key, project_id, filename_json):
        download_labelbox_json(api_key, project_id, filename_json)
        self.load_from_json(filename_json)

    def load_from_json(self, filename_json):
        self.raw_data = filter_labelbox_json(filename_json)
