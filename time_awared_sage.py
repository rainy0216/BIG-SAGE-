import asyncio
from stellargraph.data.explorer import GraphWalk
from stellargraph.layer import HinSAGE
from stellargraph.mapper import HinSAGELinkGenerator
from typing import List
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout, Reshape
import tensorflow as tf


class TimeAwaredLinkGenerator(HinSAGELinkGenerator):
    def __init__(
            self,
            G,
            batch_size,
            num_samples,
            head_node_types=None,
            schema=None,
            seed=None,
            name=None,
    ):
        super().__init__(G, batch_size, num_samples, head_node_types, schema, seed, name)
        # The sampler used to generate random samples of neighbours
        self.sampler = TimeAwaredWalk(G, graph_schema=self.schema, seed=seed)


class TimeAwaredWalk(GraphWalk):
    # def get_timestamps_adj(self):
    #     return self.graph.to_adjacency_matrix(weighted=True)
    def run(self, nodes, n_size, n=1, seed=None):
        """
        Args:
            nodes (list): A list of root node ids such that from each node n BFWs will be generated
                with the number of samples per hop specified in n_size.
            n_size (int): The number of neighbouring nodes to expand at each depth of the walk. Sampling of
            n (int, default 1): Number of walks per node id. Neighbours with replacement is always used regardless
                of the node degree and number of neighbours requested.
            seed (int, optional): Random number generator seed; default is None

        Returns:
            A list of lists such that each list element is a sequence of ids corresponding to a sampled Heterogeneous
            BFW.
        """
        self._check_sizes(n_size)
        self._check_common_parameters(nodes, n, len(n_size), seed)
        rs = self._get_random_state(seed)
        adj = self.get_adjacency_types()
        # timestamps = self.get_timestamps_adj()

        walks = []
        d = len(n_size)  # depth of search

        async def sample_step(node):
            for _ in range(n):  # do n bounded breadth first walks from each root node
                q = list()  # the queue of neighbours
                walk = list()  # the list of nodes in the subgraph of node

                node_type = self.graph.node_type(node, use_ilocs=True)
                q.extend([(node, node_type, 0)])

                # add the root node to the walks
                walk.append([node])
                while len(q) > 0:
                    # remove the top element in the queue and pop the item from the front of the list
                    frontier = q.pop(0)
                    current_node, current_node_type, depth = frontier
                    depth = depth + 1  # the depth of the neighbouring nodes

                    # consider the subgraph up to and including depth d from root node
                    if depth <= d:
                        # Find edge types for current node type
                        current_edge_types = self.graph_schema.schema[current_node_type]
                        # Create samples of neigbhours for all edge types
                        for et in current_edge_types:
                            neigh_et = adj[et][current_node]
                            weights = [self.graph._edge_weights(current_node, i, use_ilocs=True)[0] for i in neigh_et]
                            # neigh_ts = [timestamps[(current_node, i)] for i in neigh_et]
                            # If there are no neighbours of this type then we return None
                            # in the place of the nodes that would have been sampled
                            # YT update: with the new way to get neigh_et from adj[et][current_node], len(neigh_et) is always > 0.
                            # In case of no neighbours of the current node for et, neigh_et == [None],
                            # and samples automatically becomes [None]*n_size[depth-1]
                            if len(neigh_et) > 0:
                                # 采样
                                samples = rs.choices(population=neigh_et, weights=weights, k=n_size[depth - 1])
                                # samples = [neigh_et[i] for i in random_pick(neigh_ts, n_size[depth - 1])]
                            else:  # this doesn't happen anymore, see the comment above
                                _size = n_size[depth - 1]
                                samples = [-1] * _size

                            walk.append(samples)
                            q.extend(
                                [
                                    (sampled_node, et.n2, depth)
                                    for sampled_node in samples
                                ]
                            )

                # finished i-th walk from node so add it to the list of walks as a list
                walks.append(walk)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = asyncio.wait([sample_step(node) for node in nodes])  # 划分chunk
        loop.run_until_complete(tasks)
        loop.close()

        # 这里注释的是无协程的采样方式
        # for node in nodes:  # iterate over root nodes
        #     for _ in range(n):  # do n bounded breadth first walks from each root node
        #         q = list()  # the queue of neighbours
        #         walk = list()  # the list of nodes in the subgraph of node
        #
        #         # Start the walk by adding the head node, and node type to the frontier list q
        #         node_type = self.graph.node_type(node, use_ilocs=True)
        #         q.extend([(node, node_type, 0)])
        #
        #         # add the root node to the walks
        #         walk.append([node])
        #         while len(q) > 0:
        #             # remove the top element in the queue and pop the item from the front of the list
        #             frontier = q.pop(0)
        #             current_node, current_node_type, depth = frontier
        #             depth = depth + 1  # the depth of the neighbouring nodes
        #
        #             # consider the subgraph up to and including depth d from root node
        #             if depth <= d:
        #                 # Find edge types for current node type
        #                 current_edge_types = self.graph_schema.schema[current_node_type]
        #                 # Create samples of neigbhours for all edge types
        #                 for et in current_edge_types:
        #                     neigh_et = adj[et][current_node]
        #                     weights = [self.graph._edge_weights(current_node, i, use_ilocs=True)[0] for i in neigh_et]
        #                     # neigh_ts = [timestamps[(current_node, i)] for i in neigh_et]
        #                     # If there are no neighbours of this type then we return None
        #                     # in the place of the nodes that would have been sampled
        #                     # YT update: with the new way to get neigh_et from adj[et][current_node], len(neigh_et) is always > 0.
        #                     # In case of no neighbours of the current node for et, neigh_et == [None],
        #                     # and samples automatically becomes [None]*n_size[depth-1]
        #                     if len(neigh_et) > 0:
        #                         # 采样
        #                         samples = rs.choices(population=neigh_et, weights=weights, k=n_size[depth - 1])
        #                         # samples = [neigh_et[i] for i in random_pick(neigh_ts, n_size[depth - 1])]
        #                     else:  # this doesn't happen anymore, see the comment above
        #                         _size = n_size[depth - 1]
        #                         samples = [-1] * _size
        #
        #                     walk.append(samples)
        #                     q.extend(
        #                         [
        #                             (sampled_node, et.n2, depth)
        #                             for sampled_node in samples
        #                         ]
        #                     )
        #         # finished i-th walk from node so add it to the list of walks as a list
        #         walks.append(walk)

        return walks


class TimeAwaredSAGE(HinSAGE):

    def __init__(
            self,
            layer_sizes,
            generator=None,
            aggregator=None,
            bias=True,
            dropout=0.0,
            normalize="l2",
            activations=None,
            kernel_initializer="glorot_uniform",
            kernel_regularizer=None,
            kernel_constraint=None,
            bias_initializer="zeros",
            bias_regularizer=None,
            bias_constraint=None,
            n_samples=None,
            input_neighbor_tree=None,
            input_dim=None,
            multiplicity=None,
            skip_connect=True,
    ):
        self.skip_connect = skip_connect
        super().__init__(layer_sizes,
                         generator,
                         aggregator,
                         bias,
                         dropout,
                         normalize,
                         activations,
                         kernel_initializer,
                         kernel_regularizer,
                         kernel_constraint,
                         bias_initializer,
                         bias_regularizer,
                         bias_constraint,
                         n_samples,
                         input_neighbor_tree,
                         input_dim,
                         multiplicity)

    def __call__(self, xin: List):
        """
        Apply aggregator layers

        Args:
            x (list of Tensor): Batch input features

        Returns:
            Output tensor
        """

        def apply_layer(x: List, layer: int, input_layers: List):
            """
            Compute the list of output tensors for a single HinSAGE layer

            Args:
                x (List[Tensor]): Inputs to the layer
                layer (int): Layer index

            Returns:
                Outputs of applying the aggregators as a list of Tensors

            """
            layer_out = []
            for i, (node_type, neigh_indices) in enumerate(self.neigh_trees[layer]):
                # The shape of the head node is used for reshaping the neighbour inputs
                head_shape = K.int_shape(x[i])[1]

                # Aplly dropout and reshape neighbours per node per layer
                neigh_list = [
                    Dropout(self.dropout)(
                        Reshape(
                            (
                                head_shape,
                                self.n_samples[self._depths[i]],
                                self.dims[layer][self.subtree_schema[neigh_index][0]],
                            )
                        )(x[neigh_index])
                    )
                    for neigh_index in neigh_indices
                ]

                # Apply dropout to head inputs
                x_head = Dropout(self.dropout)(x[i])

                # Apply aggregator to head node and reshaped neighbour nodes
                layer_out.append(self._aggs[layer][node_type]([x_head] + neigh_list))

            if self.skip_connect:
                layer_out[0] = tf.concat([layer_out[0], input_layers[0]], axis=-1)
                layer_out[1] = tf.concat([layer_out[1], input_layers[1]], axis=-1)

            return layer_out

        # Form HinSAGE layers iteratively
        self.layer_tensors = []
        h_layer = xin
        for layer in range(0, self.n_layers):
            h_layer = apply_layer(h_layer, layer, xin[:2])
            self.layer_tensors.append(h_layer)

        # Remove neighbourhood dimension from output tensors
        # note that at this point h_layer contains the output tensor of the top (last applied) layer of the stack
        h_layer = [
            Reshape(K.int_shape(x)[2:])(x) for x in h_layer if K.int_shape(x)[1] == 1
        ]

        # Return final layer output tensor with optional normalization
        return (
            self._normalization(h_layer[0])
            if len(h_layer) == 1
            else [self._normalization(xi) for xi in h_layer]
        )
