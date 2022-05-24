import stellargraph as sg
import numpy as np
import pandas as pd

# Embedding Imports
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from stellargraph import globalvar

# Visualization Imports
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from stellargraph.mapper import GraphSAGENodeGenerator
import matplotlib.pyplot as plt

NETWORK_DIR = "data/prepared/reddit/net_00"

# Pair Generation Parameters
WALK_LENGTH = 3
WALK_NUMBER = 2

# Node Embedding Parameters
BATCH_SIZE = 100
EPOCHS = 10
NEIGHBOR_SAMPLE_SIZES = [4, 8] # TODO - Understand what this parameter is.
GRAPHSAGE_LAYER_SIZES = [4, 8] # I have small feature vectors, lets try small layers.
ADAM_LR = 1e-3


def load_dict(dict_fn):
	ret = dict()
	with open(dict_fn, 'r') as f:
		for line in f.readlines():
			k, v = line.split(", ")
			ret[k] = v
	return ret


redditid_2_netid = load_dict("{}/rid_2_netid.csv".format(NETWORK_DIR))
netid_2_redditid = load_dict("{}/netid_2_rid.csv".format(NETWORK_DIR))
subid_2_netid    = load_dict("{}/sid_2_netid.csv".format(NETWORK_DIR))
netid_2_subid    = load_dict("{}/netid_2_sid.csv".format(NETWORK_DIR))
fips_2_netid     = load_dict("{}/fips_2_netid.csv".format(NETWORK_DIR))
netid_2_fips     = load_dict("{}/netid_2_fips.csv".format(NETWORK_DIR))

edge_df = pd.read_csv("{}/edges.csv".format(NETWORK_DIR), names=["source", "target"])
nodes = np.genfromtxt("{}/nodes.csv".format(NETWORK_DIR), dtype=int, delimiter=", ")

node_labels = []
for n in range(len(nodes)):
	label = 0
	if(nodes[n][0] == 1):
		# label = "user"
		label = 1
	if(nodes[n][1] == 1):
		# label = "subreddit"
		label = 2
	if(nodes[n][2] == 1):
		# label = "county"
		label = 3
	node_labels.append(label)


graph = sg.StellarGraph(nodes, edge_df)
print(graph.info())

#### Embedding Training
## Training Pair Sampling and Generation
nodes = list(graph.nodes())
unsupervised_samples = UnsupervisedSampler(
    graph,
    nodes=nodes,
    length=WALK_LENGTH,
    number_of_walks=WALK_NUMBER)

pair_generator = GraphSAGELinkGenerator(
	graph,
	BATCH_SIZE,
	NEIGHBOR_SAMPLE_SIZES)

# This generator will provide the actual pairs of walk-co-occuring nodes
train_gen = pair_generator.flow(unsupervised_samples)

## GraphSAGE Model building
# GraphSAGE layer
layer_sage = GraphSAGE(
    layer_sizes=GRAPHSAGE_LAYER_SIZES,
    generator=pair_generator,
    bias=True,
    dropout=0.0,
    normalize="l2")
sl_x_in, sl_x_out = layer_sage.in_out_tensors()

# Link Prediction/Classification Layer
layer_linkpred = link_classification(
    output_dim=1,
    output_act="sigmoid",
    edge_embedding_method="ip")(sl_x_out)

# Stack the complete model - Note: Idk wtf keras does here.
model = keras.Model(inputs=sl_x_in, outputs=layer_linkpred)
model.compile(
	optimizer=keras.optimizers.Adam(learning_rate=ADAM_LR),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy])

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    verbose=1,
    use_multiprocessing=False,
    workers=4,
    shuffle=True,
)

### Embedding Extraction
# We want to extract the embeddings used by the above model to do
# the link prediction. That means 'recovering' the GraphSAGE outputs
# and making a new keras.Model with them.
ne_x_in  = sl_x_in[0::2] # We want every other node? idk.
ne_x_out = sl_x_out[0]     # Same, not sure wtf this is? idk.

embedding_model = keras.Model(inputs=ne_x_in, outputs=ne_x_out)

# Note - Made a change from the example here; I just reuse the same list of nodes?
#        in the example, we have actual labels (cora journal types), in this we
#        dont? So I can just say "gen node samples from the same list?" I think?
node_gen = GraphSAGENodeGenerator(
	graph,
	BATCH_SIZE,
	NEIGHBOR_SAMPLE_SIZES).flow(nodes)
node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)


### Embedding Visulaization
# TSNE to go from the embedding dimension down to 2 dimensions
X = node_embeddings
trans = TSNE(
	n_components=2,
	n_iter=1000,   # Just for testing TODO - CHANGE FOR REAL RUNS
	# angle=0.05, # JUST FOR TESTING - TODO - CHANGE TO 0.5 for real runs
	verbose=1)
emb_transformed = pd.DataFrame(trans.fit_transform(X))
emb_transformed['label'] = node_labels

alpha = 0.7
fig, ax = plt.subplots(figsize=(7, 7))
scatter = ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    c=emb_transformed['label'],
    cmap="jet",
    alpha=alpha,)

legend1 = ax.legend(
	*scatter.legend_elements(),
    loc="lower left",
    title="Node Type")
ax.add_artist(legend1)

ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title("TSNE visualization of GraphSAGE embeddings for reddit multi-layer network")
plt.savefig("test.png")
plt.show()
