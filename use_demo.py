import timm
import pickle

import torch
from PIL import Image
import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
class Model(object):
    def __init__(self):

        self.model = timm.create_model("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", pretrained=False)
        #加载本地pytorch_model.bin
        model_dict = torch.load("pytorch_model.bin",map_location=torch.device('cuda:1'))
        self.model.load_state_dict(model_dict)

        self.model = self.model.eval()
        self.model.reset_classifier(0, '')
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.cluster_model = pickle.load(open("cluster_model_unique.pkl", 'rb'))
    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        flattened = self.model(self.transforms(img).unsqueeze(0)).detach().flatten().numpy()
        flattened = flattened.astype(float)
        tmp = self.cluster_model.predict(flattened.reshape(1, -1).copy())
        if tmp == 5:
            return "dropped"
        else:
            return "not dropped"
if __name__ == '__main__':
    model = Model()
    print(model.predict("img.png"))