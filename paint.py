import matplotlib.pyplot as plt
import networkx as nx


class TreePainter(object):

    def __init__(self, input_list, output_file, id2word, title = None, partition = None):
        self.origin_list = input_list
        self.counter = 0  # 为节点命名，将和弦符号定义为其属性
        self.sequence_index = 0
        self.id2word = id2word
        self.title = title # 歌曲信息
        self.partition = partition # 音乐结构
        self.graph = nx.Graph()
        self.list_to_tree(self.origin_list, layer=0)
        self.paint_tree(output_file)



    def list_to_tree(self, input, layer):  # 递归地添加所有节点，做成中序树
        if isinstance(input, list):  # 当前节点不是叶节点
            left_node = self.list_to_tree(input[0], layer + 1)  # 左子树，返回子树的root
            current_node = self.add_node(layer, tag = None)  # 当前节点
            right_node = self.list_to_tree(input[1], layer + 1)  # 右子树，返回子树的root
            self.graph.add_edge(left_node, current_node)
            self.graph.add_edge(right_node, current_node)
        else:  # 当前节点是叶节点
            current_node = self.add_node(layer, tag=input)
        return current_node

    def add_node(self, layer, tag):
        self.graph.add_node(self.counter)
        if tag is not None:
            tag = self.id2word[int(tag.item())]
            self.graph.nodes[self.counter]['tag'] = tag
        else:
            self.graph.nodes[self.counter]['tag'] = ""
        self.graph.nodes[self.counter]['layer'] = layer
        self.graph.nodes[self.counter]['sequence_index'] = self.sequence_index
        self.counter += 1  # 更新counter标记
        self.sequence_index += 1
        return self.counter - 1  # 返回这次添加的counter_id

    def paint_tree(self, output_dir):
        max_layer = max([self.graph.nodes[k]['layer'] for k in list(self.graph.nodes)])
        sequence_len = max([self.graph.nodes[k]['sequence_index'] for k in list(self.graph.nodes)])
        try:
            fig = plt.figure(figsize=(int(max_layer / 1.5), int(sequence_len / 4)))
            nx.draw_networkx(self.graph,
                    pos = {
                        k: ( self.graph.nodes[k]['layer'], sequence_len - self.graph.nodes[k]['sequence_index'])
                        if self.graph.nodes[k]['tag'] == ""
                        else ((max_layer+1), sequence_len - self.graph.nodes[k]['sequence_index'])
                        for k in list(self.graph.nodes)
                    },
                    labels = {
                        k: self.graph.nodes[k]['tag']
                        for k in list(self.graph.nodes)
                        if self.graph.nodes[k]['tag'] != ""
                    },
                    font_size=13,
                    node_size=200,
                    node_color= 'gold',
                    edge_color= 'silver',
                    width=3,
                    )
            plt.title(self.title)
            plt.tight_layout()
            plt.savefig(output_dir)
            plt.close(fig)
        except Exception as e:
            print(e.args)