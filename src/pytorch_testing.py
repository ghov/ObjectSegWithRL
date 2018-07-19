from ObjectSegWithRL.src.greg_cnn import GregNet
#from ObjectSegWithRL.src.main_rlnn import GregNet
import torch

#my_alex = AlexNet()
my_greg = GregNet(15)
#my_greg = GregNet(4, 80)
#print(torch.zeros(3, 10, 10))

greg_out = my_greg.forward(torch.zeros((3, 224, 224)).unsqueeze_(0))

print(greg_out.size())

#greg_out = my_greg.forward(alex_out)

#print(greg_out.size())

#temp_list = (5,8,80)

#split_out = torch.split(greg_out, temp_list, dim=1)

#actions, amounts, classes = split_out

# print(actions.size())
# print(amounts.size())
# print(classes.size())
#
# soft = torch.nn.Softmax(dim=1)
# print(soft(actions))
# print(soft(classes))

test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])










#def mixed_loss(old_iou, new_iou, classification_vector):





# def new_polygon(old_polygon_list, selected_index_int, change_amounts_tuple_int):
#     """
#
#     :param old_polygon_list: a list of x1,y1,x2,y2,... describing the polygon. This should be in matplotlib?
#     :param selected_index_int: the index of the point that the agent chose to move
#     :param change_amounts_tuple_int: the amount of change in x and y for the chosen point
#     :return: the new list, in same format as the input list. This should be in matplotlib?
#     """
#
#     change_x, change_y = change_amounts_tuple_int
#     new_polygon_list = old_polygon_list.copy()
#     new_polygon_list[selected_index_int * 2] = old_polygon_list[selected_index_int * 2] + change_x
#     new_polygon_list[(selected_index_int * 2) + 1] = old_polygon_list[(selected_index_int *2) + 1] + change_y
#
#     return new_polygon_list


# Next, need to figure out how to do backprop


