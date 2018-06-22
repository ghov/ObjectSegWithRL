from torch.utils.data import DataLoader

# The images are not of the same dimensions. Will need to resize all images.







# This will define the 4 (or n) point mask applied to the image.
class mask_state():

# The state will be represented as 8 points in a vector. It will go as: x1, y1, x2, y2, x3, y3, x4, y4
    def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.x4 = x4
        self.y4 = y4


    def to_tensor(self):



    def to_polygon(self):




    def to_binary_mask(self):





def combine_cnn_with_state(cnn_feature_vector, state_vector):
    # Takes as input the feature vector from the cnn and combines it with the feature vector from the state





def get_reward(old_iou, new_iou):
    # Takes as input the intersection over union at time t and t+1. Returns +sign(new_iou - old_iou)




