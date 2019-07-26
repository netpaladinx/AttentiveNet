import os
import copy
import random

import yaml
import networkx as nx
import graphviz as gv


GRAPH_DIR = '../anet_graphs'
os.makedirs(GRAPH_DIR, exist_ok=True)


class RandomGraph(object):
    def __init__(self, rg_model, n_nodes, er_p=None, ba_m=None, ws_k=None, ws_p=None, seed=None, n_tries=100):
        self.rg_model = rg_model
        self.n_nodes = n_nodes
        self.er_p = er_p
        self.ba_m = ba_m
        self.ws_k = ws_k
        self.ws_p = ws_p
        self.seed = seed

        self.rg_graph = self._build_random_graph(n_tries)
        self.dag_edges = self._get_dag_edges(self.rg_graph)
        self.inputs, self.outputs = self._get_inputs_and_outputs(self.dag_edges)

    def _build_random_graph(self, n_tries):
        for _ in range(n_tries):
            if self.rg_model == 'erdos_renyi':
                graph = nx.erdos_renyi_graph(self.n_nodes, self.er_p, seed=self.seed)
            elif self.rg_model == 'barabasi_albert':
                graph = nx.barabasi_albert_graph(self.n_nodes, self.ba_m, seed=self.seed)
            elif self.rg_model == 'watts_strogatz':
                graph = nx.watts_strogatz_graph(self.n_nodes, self.ws_k, self.ws_p, seed=self.seed)
            else:
                raise nx.NetworkXError('Invalid `rg_model`: {}'.format(self.rg_model))
            if nx.is_connected(graph):
                return graph
        raise nx.NetworkXError('Maximum number of tries ({:d}) exceeded'.format(n_tries))

    def _get_dag_edges(self, rg_graph):
        # nodes are one-indexed
        return [(e[0] + 1, e[1] + 1) for e in sorted(list(rg_graph.edges), key=lambda e: (e[0] + 1, e[1] + 1))]

    def _get_inputs_and_outputs(self, edges):
        starts, ends = zip(*edges)
        starts = set(starts)
        ends = set(ends)
        inputs = sorted(list(starts - ends))
        outputs = sorted(list(ends - starts))
        return inputs, outputs


class Node(object):
    def __init__(self, i, stage_graph):
        self.id = i
        self.stage_graph = stage_graph
        self.is_master_input = False
        self.is_master_output = False
        self.is_sub_input = False
        self.is_sub_output = False
        self.in_neighbors = []
        self.out_neighbors = []

    @property
    def in_nei_ids(self):
        return [nei.id for nei in self.in_neighbors]

    @property
    def out_nei_ids(self):
        return [nei.id for nei in self.out_neighbors]

    @property
    def n_in_neis(self):
        return len(self.in_neighbors)

    @property
    def n_out_neis(self):
        return len(self.out_neighbors)


class StageGraph(object):
    def __init__(self, stage_dct):
        self.id = stage_dct['_stage']
        self.stage_dct = stage_dct

        self.n_nodes = stage_dct['n_internal_nodes'] + 2
        self.nodes = [Node(i, self) for i in range(self.n_nodes)]

        self.master_input = self.nodes[0]
        self.master_output = self.nodes[-1]
        self.master_input.is_master_input = True
        self.master_output.is_master_output = True

        self.sub_inputs = [self.nodes[i] for i in stage_dct['inputs']]
        self.sub_outputs = [self.nodes[i] for i in stage_dct['outputs']]
        
        for node in self.sub_inputs:
            node.is_sub_input = True
            node.in_neighbors.append(self.master_input)
            self.master_input.out_neighbors.append(node)
            
        for node in self.sub_outputs:
            node.is_sub_output = True
            node.out_neighbors.append(self.master_output)
            self.master_output.in_neighbors.append(node)

        for e in stage_dct.get('edges', []):
            n0 = self.nodes[e[0]]
            n1 = self.nodes[e[1]]
            n0.out_neighbors.append(n1)
            n1.in_neighbors.append(n0)

        for node in self.nodes:
            node.in_neighbors.sort(key=lambda nd: nd.id)
            node.out_neighbors.sort(key=lambda nd: nd.id)

    def __getattr__(self, name):
        if name in self.stage_dct:
            return self.stage_dct[name]
        else:
            raise AttributeError('Has no attribute `{}`'.format(name))


class ANetGraph(object):
    def __init__(self, name):
        self.name = name
        graph_filepath = self.get_or_build_graph(name)
        self.stage_graphs = self._load_graph(graph_filepath)

    def _load_graph(self, graph_filepath):
        with open(graph_filepath) as f:
            graph_dct = yaml.load(f, Loader=yaml.Loader)
        return [StageGraph(stage_dct) for stage_dct in graph_dct['graph']]

    def get_or_build_graph(self, name):
        config_filepath = os.path.join(GRAPH_DIR, name + '.config.yaml')
        data_filepath = os.path.join(GRAPH_DIR, name + '.data.yaml')

        if os.path.exists(data_filepath):
            return data_filepath
        if not os.path.exists(config_filepath):
            raise FileNotFoundError('File not found at: %s' % config_filepath)

        with open(config_filepath) as config_file:
            graph_config = yaml.load(config_file, Loader=yaml.Loader)

        seed = graph_config['seed']
        random.seed(a=seed)
        a, b = 0, 99999

        graph_data = copy.deepcopy(graph_config)
        for stage in graph_data['graph']:
            if stage['type'] == 'randomly_wired':
                if stage['rw_graph'] == 'erdos_renyi':
                    seed = random.randint(a, b)
                    rg = RandomGraph('erdos_renyi', stage['n_internal_nodes'], er_p=stage['er_p'], seed=seed)
                    stage['seed'] = seed
                    stage['inputs'] = rg.inputs
                    stage['outputs'] = rg.outputs
                    stage['edges'] = rg.dag_edges
                elif stage['rw_graph'] == 'barabasi_albert':
                    seed = random.randint(a, b)
                    rg = RandomGraph('barabasi_albert', stage['n_internal_nodes'], ba_m=stage['ba_m'], seed=seed)
                    stage['seed'] = seed
                    stage['inputs'] = rg.inputs
                    stage['outputs'] = rg.outputs
                    stage['edges'] = rg.dag_edges
                elif stage['rw_graph'] == 'watts_strogatz':
                    seed = random.randint(a, b)
                    rg = RandomGraph('watts_strogatz', stage['n_internal_nodes'], ws_k=stage['ws_k'], ws_p=stage['ws_p'], seed=seed)
                    stage['seed'] = seed
                    stage['inputs'] = rg.inputs
                    stage['outputs'] = rg.outputs
                    stage['edges'] = rg.dag_edges
            else:
                stage['inputs'] = [1]
                stage['outputs'] = [1]
                stage['n_internal_nodes'] = 1

        class _Dumper(yaml.Dumper):
            def increase_indent(self, flow=False, indentless=False):
                return super(_Dumper, self).increase_indent(flow, False)

        with open(data_filepath, 'w') as data_file:
            yaml.dump(graph_data, data_file, default_flow_style=False, Dumper=_Dumper)
        return data_filepath

    def draw_graph(self, name=None):
        if name is None:
            name = self.name
        data_filepath = os.path.join(GRAPH_DIR, name + '.data.yaml')

        if not os.path.exists(data_filepath):
            raise FileNotFoundError('File not found at: %s' % data_filepath)

        with open(data_filepath) as data_file:
            graph_data = yaml.load(data_file, Loader=yaml.Loader)

        stage_nodes = []
        for i, stage in enumerate(graph_data['graph']):
            stage_id = stage['_stage']
            if i == 0:
                stage_nodes.append('%d:i' % stage_id)
            else:
                pre_stage_id = graph_data['graph'][i - 1]['_stage']
                stage_nodes.append('%d:o/%d:i' % (pre_stage_id, stage_id))
            if i == len(graph_data['graph']) - 1:
                stage_nodes.append('%d:o' % stage_id)

        all_edges = []
        stage_nodes = set()
        for i, stage in enumerate(graph_data['graph']):
            stage_id = stage['_stage']
            last_stage_id = graph_data['graph'][i - 1]['_stage'] if i > 0 else None
            next_stage_id = graph_data['graph'][i + 1]['_stage'] if i < len(graph_data['graph']) - 1 else None
            start_node = '%d-O/%d-I' % (last_stage_id, stage_id) if last_stage_id is not None else '%d-I' % stage_id
            end_node = '%d-O/%d-I' % (stage_id, next_stage_id) if next_stage_id is not None else '%d-O' % stage_id
            stage_nodes.add(start_node)
            stage_nodes.add(end_node)

            for v_id in stage['inputs']:
                node = '%d-%d' % (stage_id, v_id)
                all_edges.append((start_node, node))

            if 'edges' in stage:
                for v1_id, v2_id in stage['edges']:
                    node1 = '%d-%d' % (stage_id, v1_id)
                    node2 = '%d-%d' % (stage_id, v2_id)
                    all_edges.append((node1, node2))

            for v_id in stage['outputs']:
                node = '%d-%d' % (stage_id, v_id)
                all_edges.append((node, end_node))

        di_graph = gv.Digraph()
        di_graph.attr(kw='graph', rankdir='BT')

        di_graph.attr('node', shape='box', style='filled', color='lightgrey')
        for node in stage_nodes:
            di_graph.node(node)
        di_graph.attr('node', shape='ellipse', style='', color='black')

        for node1, node2 in all_edges:
            di_graph.edge(node1, node2)

        fig_filepath = os.path.join(GRAPH_DIR, name + '.gv')
        di_graph.render(fig_filepath, view=True)


if __name__ == '__main__':
    anet_graph = ANetGraph('anet_small_ws_v1_1234')
    anet_graph.draw_graph()
