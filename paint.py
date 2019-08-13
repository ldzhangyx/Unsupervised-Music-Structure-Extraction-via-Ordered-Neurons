import matplotlib.pyplot as plt
import torch
import networkx as nx
import pretty_midi


class TreePainter(object):

    def __init__(self, input_list, output_file, id2word, title=None, partition=None):
        self.origin_list = input_list
        self.counter = 0  # 为节点命名，将和弦符号定义为其属性
        self.sequence_index = 0
        self.id2word = id2word
        self.title = title  # 歌曲信息
        self.partition = partition  # 音乐结构
        self.vertical = False
        self.color = 'positive'  # color tags > 0
        self.depth_limit = 5
        self.reschedule_index = True
        self.graph = nx.Graph()
        self.note_name = True
        self.ignore_sustain = True
        self.midi = True
        self.is_rewrite = True

        # Action
        if self.midi:
            self.create_midi(self.origin_list, output_file)
        if self.is_rewrite:
            _sustain = torch.tensor(self.id2word.index('0'), dtype=torch.float32)
            _rest = torch.tensor(self.id2word.index('-1'), dtype=torch.float32)
            self.origin_list, _ = self.rewrite(self.origin_list, False, _rest, _sustain)
        if self.ignore_sustain:
            self.origin_list = self.reshape_list(self.origin_list)
        self.list_to_tree(self.origin_list, layer=0)
        self.paint_tree(output_file)

    def create_midi(self, input_list, output_file):
        midi_file = pretty_midi.PrettyMIDI()
        cello_program = pretty_midi.instrument_name_to_program('acousticgrandpiano')
        cello = pretty_midi.Instrument(program=cello_program)
        _mel_list = self.flat_list(self.origin_list) # 是一个tensor list
        current_time = 0
        timespan = 0
        current_pitch = None
        _sustain = torch.tensor(self.id2word.index('0'), dtype=torch.float32)
        _rest = torch.tensor(self.id2word.index('-1'), dtype=torch.float32)
        for element in _mel_list:
            if element == _rest:
                continue
            elif element == _sustain:
                timespan += 0.25
            else:
                if timespan > 0:
                    cello.notes.append(
                        pretty_midi.Note(
                            velocity=100,
                            start=current_time,
                            end = current_time + timespan,
                            pitch = int(current_pitch)
                        )
                    )
                current_pitch = element
        midi_file.instruments.append(cello)
        save_file = output_file[-4:] + '.mid'
        midi_file.write(save_file)

    def rewrite(self, node, flag, _rest, _sustain):
        if isinstance(node, list):
            node[0], flag = self.rewrite(node[0], flag, _rest, _sustain)
            node[1], flag = self.rewrite(node[1], flag, _rest, _sustain)
        else:
            if node == _rest and not flag: # 开启一段连续的rest
                flag = True
            elif node == _rest and flag: # 连续rest中间
                node = _sustain
            elif node != _rest:
                flag = False # rest结束
        return node, flag

    def reshape_list(self, input_list):
        _sustain = torch.tensor(self.id2word.index('0'), dtype=torch.float32)
        _rest = torch.tensor(self.id2word.index('-1'), dtype=torch.float32)
        # rest由于数据处理的时候脑子进水，rest需要置0，但仅需要第一次置零即可。
        if isinstance(input_list[0], list): # 递归地解决左子树
            input_list[0] = self.reshape_list(input_list[0])
        if isinstance(input_list[1], list): # 递归地解决右子树
            input_list[1] = self.reshape_list(input_list[1])
        # 边界有几种情况：都不删，都删，和只删一边
        if not isinstance(input_list[0], list) and not isinstance(input_list[1], list): # 两个子树都为叶子节点，边界条件
            if input_list[0] == _sustain and input_list[1] == _sustain:  # 都删需要满足两边都是叶子
                return _sustain
        if not isinstance(input_list[0], list):
            if input_list[0] == _sustain: # 删左边需要保证左边是叶子
                return input_list[1]
        if not isinstance(input_list[1], list):
            if input_list[1] == _sustain: # 删右边需要保证右边是叶子
                return input_list[0]
        return input_list


    def flat_list(self, the_list):
        now = the_list[:]
        res = []
        while now:
            head = now.pop(0)
            if isinstance(head, list):
                now = head + now
            else:
                res.append(head)
        return res

    def list_to_tree(self, input, layer):  # 递归地添加所有节点，做成中序树
        if isinstance(input, list) and layer < self.depth_limit:  # 当前节点不是叶节点 ，或深度没有到达限制
            left_node = self.list_to_tree(input[0], layer + 1)  # 左子树，返回子树的root
            current_node = self.add_node(layer, tag=None)  # 当前节点
            right_node = self.list_to_tree(input[1], layer + 1)  # 右子树，返回子树的root
            self.graph.add_edge(left_node, current_node)
            self.graph.add_edge(right_node, current_node)
        elif not isinstance(input, list):  # 当前节点是叶节点
            current_node = self.add_node(layer, tag=input)
        elif layer >= self.depth_limit:
            child_list = self.flat_list(input)
            left_list = child_list[:int(len(child_list) / 2)]
            right_list = child_list[int(len(child_list) / 2):]
            child_node_list = list()
            for leaf in left_list:  # 左子树
                child_node = self.list_to_tree(leaf, layer + 1)
                child_node_list.append(child_node)
            current_node = self.add_node(layer, tag=None)  # 多叉树，但还是要保持中序（为了美观）
            for leaf in right_list:  # 右子树
                child_node = self.list_to_tree(leaf, layer + 1)
                child_node_list.append(child_node)
            for child_node in child_node_list:
                self.graph.add_edge(child_node, current_node)
        return current_node

    def add_node(self, layer, tag):
        self.graph.add_node(self.counter)
        if tag is not None:  # 叶子
            tag = self.id2word[int(tag.item())]
            if self.note_name:
                if int(tag) > 0:
                    tag = pretty_midi.note_number_to_name(int(tag))
            self.graph.nodes[self.counter]['tag'] = tag

        else:  # 中间节点
            self.graph.nodes[self.counter]['tag'] = ""
        self.graph.nodes[self.counter]['layer'] = layer
        self.graph.nodes[self.counter]['sequence_index'] = self.sequence_index
        self.counter += 1  # 更新counter标记
        self.sequence_index += 1
        return self.counter - 1  # 返回这次添加的counter_id

    def paint_tree(self, output_dir):
        max_layer = max([self.graph.nodes[k]['layer'] for k in list(self.graph.nodes)])
        sequence_len = max([self.graph.nodes[k]['sequence_index'] for k in list(self.graph.nodes)]) + 1
        leaf_len = len([self.graph.nodes[k] for k in list(self.graph.nodes) if self.graph.nodes[k]['tag'] != ''])

        if self.reschedule_index:  # 将子树尝试均匀排列
            position_counter = 0
            beat_flag = 4
            blank_counter = 0
            bar_flag = leaf_len / 4
            for leaf_index in range(sequence_len):  # 先修正最底层
                if self.graph.nodes[leaf_index]['tag'] != '':
                    self.graph.nodes[leaf_index]['sequence_index'] = position_counter
                    position_counter += 1
                    if (position_counter - blank_counter) % beat_flag == 0:
                        position_counter += 1
                        blank_counter += 1
                    if (position_counter - blank_counter) % bar_flag == 0:
                        position_counter += 1
                        blank_counter += 1
            # 反过来修正上面所有层数
            for layer_index in range(max_layer):  # 0 to max_layer - 1
                current_layer = max_layer - layer_index  # reverse
                # 寻找直接子节点
                for node_index in range(sequence_len):
                    if self.graph.nodes[node_index]['layer'] == current_layer and self.graph.nodes[node_index][
                        'tag'] == '':  # 中间节点
                        position_list = list()
                        adj_node_list = list(self.graph.adj[node_index].keys())
                        positions = [self.graph.nodes[node]['sequence_index'] for node in adj_node_list
                                     if self.graph.nodes[node]['layer'] == current_layer + 1]
                        self.graph.nodes[node_index]['sequence_index'] = sum(positions) / len(positions)

        try:
            fig = plt.figure(figsize=(int(sequence_len / 4), int(max_layer)))
            pic = nx.draw_networkx(self.graph,
                                   pos={
                                       k: (self.graph.nodes[k]['sequence_index'],
                                           (max_layer + 1) - self.graph.nodes[k]['layer'])
                                       if self.graph.nodes[k]['tag'] == ""
                                       else (self.graph.nodes[k]['sequence_index'], 0)
                                       for k in list(self.graph.nodes)
                                   },
                                   labels={
                                       k: self.graph.nodes[k]['tag']
                                       for k in list(self.graph.nodes)
                                       if self.graph.nodes[k]['tag'] != ""
                                   },
                                   font_size=13,
                                   node_size=250,
                                   node_color='gold',
                                   edge_color='silver',
                                   width=3,
                                   )

            if self.vertical:
                for _, item in pic.item():
                    item.set_rotation('vertical')

            plt.title(self.title)
            plt.tight_layout()
            plt.savefig(output_dir)
            plt.close(fig)
        except Exception as e:
            print(e.args)
