import stellargraph as sg
import numpy as np
import pandas as pd


from stellargraph.mapper import GraphSAGELinkGenerator, GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.data import UnsupervisedSampler
from tensorflow import keras
from sklearn.manifold import TSNE


def learn_embeddings(nodes,
                     edges,
                     walk_length=2,
                     walk_number=2,
                     batch_size=100,
                     neighbor_sample_sizes=None,
                     graphsage_layer_sizes=None,
                     adam_lr=1e-3,
                     num_epochs=1,
                     verbose=True):

    if neighbor_sample_sizes is None:
        neighbor_sample_sizes = [2, 3]
    if graphsage_layer_sizes is None:
        graphsage_layer_sizes = [2, 4]

    node_labels = []
    for n in range(len(nodes)):
        label = 0
        if (nodes[n][0] == 1):
            # label = "user"
            label = 1
        if (nodes[n][1] == 1):
            # label = "subreddit"
            label = 2
        if (nodes[n][2] == 1):
            # label = "county"
            label = 3
        node_labels.append(label)

    nodes_np = np.array(nodes)
    edge_df = pd.DataFrame(edges, columns=['source', 'target'])
    graph = sg.StellarGraph(nodes_np, edge_df)
    if verbose: print("stellargraph object created.")

    nodes = list(graph.nodes())
    unsupervised_samples = UnsupervisedSampler(
        graph,
        nodes=nodes,
        length=walk_length,
        number_of_walks=walk_number)

    pair_generator = GraphSAGELinkGenerator(
        graph,
        batch_size,
        neighbor_sample_sizes)
    if verbose: print("Link sampler, generator created.")

    train_gen = pair_generator.flow(unsupervised_samples)

    # GraphSAGE Model building
    # GraphSAGE layer
    layer_sage = GraphSAGE(
        layer_sizes=graphsage_layer_sizes,
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
    if verbose: print("compiling keras model.")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=adam_lr),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy])

    if verbose: print("training link prediction model.")
    history = model.fit(
        train_gen,
        epochs=num_epochs,
        verbose=1,
        use_multiprocessing=False,
        workers=4,
        shuffle=True,
    )

    ### Embedding Extraction
    # We want to extract the embeddings used by the above model to do
    # the link prediction. That means 'recovering' the GraphSAGE outputs
    # and making a new keras.Model with them.
    ne_x_in = sl_x_in[0::2]  # We want every other node? idk.
    ne_x_out = sl_x_out[0]  # Same, not sure wtf this is? idk.

    embedding_model = keras.Model(inputs=ne_x_in, outputs=ne_x_out)

    # Note - Made a change from the example here; I just reuse the same list of nodes?
    #        in the example, we have actual labels (cora journal types), in this we
    #        dont? So I can just say "gen node samples from the same list?" I think?
    node_gen = GraphSAGENodeGenerator(
        graph,
        batch_size,
        neighbor_sample_sizes).flow(nodes)
    if verbose: print("recovering node embeddings.")
    node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

    return node_embeddings, node_labels


def tsne(X):
    trans = TSNE(
        n_components=2,
        n_iter=100,  # Just for testing TODO - CHANGE FOR REAL RUNS
        # angle=0.05, # JUST FOR TESTING - TODO - CHANGE TO 0.5 for real runs
        verbose=1)
    return pd.DataFrame(trans.fit_transform(X))
