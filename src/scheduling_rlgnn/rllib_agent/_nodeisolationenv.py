import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython import display


class NodeIsolationEnv(gym.Env):
    """
    A Reinforcement Learning environment where the goal is to isolate a
    "target" node in a graph by removing its neighbors.

    The observation space is a dictionary with "x" (node features) and
    "edge_index" (graph connectivity).
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, num_nodes=20, p=0.25, render_mode=None):
        super().__init__()

        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        )
        self.render_mode = render_mode

        if self.render_mode == "human":
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.num_nodes = num_nodes
        self.p = p

        # Action space: select a node to remove.
        self.action_space = spaces.Discrete(self.num_nodes)
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=0, high=1, shape=(self.num_nodes, 1), dtype=np.float32
                ),
                # "edge_index": Graph connectivity in COO format [2, E]
                # The number of edges E is variable, so we define the space
                # with the maximum possible number of edges.
                "edge_index": spaces.Box(
                    low=0,
                    high=self.num_nodes - 1,
                    shape=(2, self.num_nodes * (self.num_nodes - 1)),
                    dtype=np.int64,
                ),
                "action_mask": spaces.MultiBinary(self.num_nodes),
            }
        )

        self.nx_graph: nx.Graph = nx.Graph()
        self.terrorist_node: int | None = None
        self.pos = None
        self.action_mask = np.ones(self.num_nodes, dtype=np.int8)

    def _get_obs(self):
        """
        Constructs the observation dictionary with "x" and "edge_index".
        """
        # "x" -> Node features [N, 1]
        node_features = np.zeros((self.num_nodes, 1), dtype=np.float32)
        if self.terrorist_node is not None:
            node_features[self.terrorist_node, 0] = 1.0

        # "edge_index" -> COO format [2, E]
        if self.nx_graph.number_of_edges() == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)
        else:
            # Transpose the edge list to get the [2, E] shape
            edge_index = np.array(
                list(self.nx_graph.edges()), dtype=np.int32
            ).T

        action_mask = np.zeros(self.num_nodes, dtype=np.int8)
        if self.nx_graph.number_of_nodes() > 0:
            for node in self.nx_graph.nodes():
                if node != self.terrorist_node:
                    action_mask[node] = 1

        return {
            "x": node_features,
            "edge_index": edge_index,
            "action_mask": action_mask,
        }

    def _get_info(self):
        return {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        while True:
            self.nx_graph = nx.erdos_renyi_graph(
                self.num_nodes, self.p, seed=self.np_random
            )
            if (
                nx.is_connected(self.nx_graph)
                and self.nx_graph.number_of_nodes() > 1
            ):
                break

        self.terrorist_node = self.np_random.choice(
            list(self.nx_graph.nodes())
        )
        self.action_mask = np.ones(self.num_nodes, dtype=np.int8)
        self.action_mask[self.terrorist_node] = 0
        # Pass a numpy.random.RandomState object to nx.spring_layout
        self.pos = nx.spring_layout(self.nx_graph, seed=seed)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        is_valid_action = (
            self.nx_graph.number_of_nodes() > 0
            and action in self.nx_graph.nodes()
            and action != self.terrorist_node
        )
        num_remaining_nodes = self.nx_graph.number_of_nodes()
        if is_valid_action:
            self.nx_graph.remove_node(action)
            reward = -1
            self.action_mask[action] = 0
        else:
            reward = -num_remaining_nodes
        observation = self._get_obs()
        info = self._get_info()
        if (
            self.terrorist_node is not None
            and self.terrorist_node in self.nx_graph.nodes()
        ):
            terminated = (num_remaining_nodes <= 2) or (
                len(observation["action_mask"].nonzero()[0]) == 0
            )
        else:
            terminated = True

        truncated = (num_remaining_nodes <= 2) or (
            len(observation["action_mask"].nonzero()[0]) == 0
        )

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        self.ax.clear()
        node_colors = [
            "red" if node == self.terrorist_node else "skyblue"
            for node in self.nx_graph.nodes()
        ]

        nx.draw(
            self.nx_graph,
            pos=self.pos,
            ax=self.ax,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            font_color="white",
            font_weight="bold",
        )

        self.ax.set_title(f"Isolate the Red Node ({self.terrorist_node})")
        display.display(self.fig)

    def close(self):
        if self.render_mode == "human":
            plt.close(self.fig)
