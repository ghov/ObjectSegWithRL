from torch.utils.data import Dataset
import json
from skimage.io import imread
import numpy as np

class GregDataset(Dataset):

    def __init__(self, annotation_file_path, root_dir_path, transform=None):

        self.transform = transform
        self.annotations = None
        self.root_dir = root_dir_path
        self.indexer = None

        with open(annotation_file_path, 'r') as read_file:
            self.annotations = json.load(read_file)

        # Store the segmentation ids in a list as strings
        self.indexer = list(self.annotations.keys())

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        seg_id = self.indexer[idx]
        image_path = self.root_dir + seg_id + '.jpg'
        temp_img = imread(image_path)

        # Check if it is greyscale
        if len(temp_img.shape) == 2:
            height, width = temp_img.shape
            temp_img = np.resize(temp_img, (height, width, 3))

        poly = self.annotations[seg_id]
        poly_np = np.asarray(poly)
        poly_ann = poly_np.astype('float').reshape(-1, len(poly))

        print(seg_id)
        # Need to add access to a transform method
        if(self.transform):
            temp_img = self.transform(temp_img)
            #poly_ann = self.transform(poly_ann)


        return temp_img, poly_ann


def main():
    annotation_file_path = '/media/greghovhannisyan/BackupData1/mscoco/annotations/by_vertex/30_vertex_poly_adjusted.json'
    root_dir_path = '/media/greghovhannisyan/BackupData1/mscoco/images/by_vertex/30/'


    g = GregDataset(annotation_file_path, root_dir_path)

    sample_img, sample_ann = g[0]
    print(sample_img.shape, sample_ann.shape)
    print(sample_ann)

    #print(len(g))


if __name__ == '__main__':
    main()