import os
import statistics
import pickle


class TimeSpanTree(object):
    def __init__(self):
        self.label = None
        self.left = None
        self.right = None
        self.is_leaf = True

def tree_editing_distance(output_list, gold_list):
    output_tree = (list_to_zsstree(output_list))
    gold_tree = (list_to_zsstree(gold_list))
    return simple_distance(output_tree, gold_tree)

def node_match_algorithm(output_tree, gold_tree):
    total_nodes = len(output_tree)
    match_nodes = 0
    for node in output_tree:
        for gold_node in gold_tree:
            if node.is_leaf:
                continue
            if node.left == gold_node.left and node.right == gold_node.right:
                match_nodes += 1
    return match_nodes/total_nodes

import os
import xml.etree.ElementTree as ET

def tstree_to_list(input_folder, output_file):
    file_list = os.listdir(input_folder)
    element_list = list()
    for file in file_list:
        file_path = '/'.join([input_folder, file])
        tree = ET.parse(file_path).getroot().find('ts') # element 'ts'
        element = xml_parse(tree)
        element_list.append(element)
    return


def xml_parse(xml):
    childs = [item.tag for item in xml]
    if 'primary' not in childs and 'secondary' not in childs: #leaf
        return 0
    else:
        return [
            xml_parse(xml.find('primary').find('ts')),
            xml_parse(xml.find('secondary').find('ts'))
        ]

# tstree_to_list("/gpfsnyu/home/yz6492/on-lstm/data/gttm/TSTree/")

def list_to_tstree(input_list, is_label = False):
    '''Give each node include leaf node an order.

    :param input_list:
    :param is_label:
    :return:
    '''

    order_count = 0

from zss import simple_distance, Node

def list_to_zsstree(input_list, label = '0'):
    if not isinstance(input_list, list): # 叶节点
        return Node(label)
    else: # 中间节点
        return Node(label)\
            .addkid(list_to_zsstree(input_list[0]))\
            .addkid(list_to_zsstree(input_list[1]))


def model_cross_validator(model, data_list):
    for i, batch in enumerate(data_list):
        train_data = data_list[:i].extend(data_list[i+1:])


def file_to_list(input_file):
    with open(input_file, 'r') as f:
        elements = f.readlines()
    def element_to_list(line_index):
        if elements[line_index].split()[2] == 'Leaf': # 叶节点
            return int(elements[line_index].split()[8]) # Leaf index
        else:
            return [element_to_list(int(elements[line_index].split()[8])),
                    element_to_list(int(elements[line_index].split()[9]))]
    tree_list= element_to_list(0)
    return tree_list



def baseline_zss_score_wrapper(model_folder, label_folder): # 计算baseline的zss distance
    output_list = os.listdir(model_folder) # len: 295，300 samples except [21, 155, 283, 289, 297] @ Eita Nakamura
    output_list = [element for element in output_list if element.endswith('.txt')]
    score_list = list()
    for file_index in output_list:
        index = file_index.split('_')[0]
        output_file = '/'.join([model_folder, file_index])
        label_file = '/'.join([label_folder, index, '{}_TS.txt'.format(index)])
        output_tree_list, label_list = file_to_list(output_file), file_to_list(label_file)
        score = tree_editing_distance(output_tree_list, label_list)
        #  print(score)
        score_list.append(score)
    score_avg = statistics.mean(score_list)
    print(score_list)
    print(score_avg)
    return score_avg

def dict_zss_score_wrapper(model_dict_pkl, label_dict_pkl):
    with open(model_dict_pkl, 'rb') as f:
        model_dict = pickle.load(f)
    with open(label_dict_pkl, 'rb') as f:
        label_dict = pickle.load(f)
    score_list = list()
    for key in model_dict:
        model_tree = model_dict[key]
        label_tree = label_dict[key]
        score = tree_editing_distance(model_tree, label_tree)
        print(score)
        score_list.append(score)
    score_avg = statistics.mean(score_list)
    print(score_list)
    print('Average Score: {}'.format(score_avg))

model_dict_pkl = "/gpfsnyu/home/yz6492/on-lstm/output/gttm/finetuned_layer_1.pkl"
label_dict_pkl = "/gpfsnyu/home/yz6492/on-lstm/data/gttm/label_zss_tree_index.pkl"
dict_zss_score_wrapper(model_dict_pkl, label_dict_pkl)

model_dict_pkl = "/gpfsnyu/home/yz6492/on-lstm/output/gttm/finetuned_layer_2.pkl"
label_dict_pkl = "/gpfsnyu/home/yz6492/on-lstm/data/gttm/label_zss_tree_index.pkl"
dict_zss_score_wrapper(model_dict_pkl, label_dict_pkl)

model_dict_pkl = "/gpfsnyu/home/yz6492/on-lstm/output/gttm/finetuned_layer_3.pkl"
label_dict_pkl = "/gpfsnyu/home/yz6492/on-lstm/data/gttm/label_zss_tree_index.pkl"
dict_zss_score_wrapper(model_dict_pkl, label_dict_pkl)

model_dict_pkl = "/gpfsnyu/home/yz6492/on-lstm/output/gttm/finetuned_layer_mean.pkl"
label_dict_pkl = "/gpfsnyu/home/yz6492/on-lstm/data/gttm/label_zss_tree_index.pkl"
dict_zss_score_wrapper(model_dict_pkl, label_dict_pkl)


# model_folder = "/gpfsnyu/home/yz6492/on-lstm/data/gttm/pgttm/0_GibbsEMUnSupervised/"
# label_folder = "/gpfsnyu/home/yz6492/on-lstm/data/gttm/pgttm/HamanakaData/"
# baseline_zss_score_wrapper(model_folder, label_folder)
import pickle

def zss_label_wrapper(input_folder, label_file): # 计算baseline的zss distance
    output_list = os.listdir(input_folder) # len: 295，300 samples except [21, 155, 283, 289, 297] @ Eita Nakamura
    output_list = [i for i in output_list if '_' not in i and i not in ['21', '155', '283', '289', '297']]
    label_dict = dict()
    for file_index in output_list:
        id = int(file_index)
        label = '/'.join([input_folder, file_index, '{}_TS.txt'.format(id)])
        label_list = file_to_list(label)
        label_dict[id] = label_list
    with open(label_file, 'wb') as f:
        pickle.dump(label_dict, f)

# input_folder = "/gpfsnyu/home/yz6492/on-lstm/data/gttm/pgttm/HamanakaData/"
# label_file = "/gpfsnyu/home/yz6492/on-lstm/data/gttm/label_zss_tree_index.pkl"
# zss_label_wrapper(input_folder, label_file)
