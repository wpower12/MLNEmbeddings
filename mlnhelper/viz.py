import matplotlib.pyplot as plt


def save_embedding_viz(emb_transformed, fn):
    alpha = 0.7
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        emb_transformed[0],
        emb_transformed[1],
        c=emb_transformed['label'],
        cmap="jet",
        alpha=alpha, )

    legend1 = ax.legend(
        *scatter.legend_elements(),
        loc="lower left",
        title="Node Type")
    ax.add_artist(legend1)

    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    plt.title("TSNE visualization of GraphSAGE embeddings for reddit multi-layer network")
    plt.savefig(fn)
