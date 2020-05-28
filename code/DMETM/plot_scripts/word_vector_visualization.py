from sklearn.manifold import TSNE

######Create a plot 2D for visualizing word vectors######

corpus = [] #Add here the nearest neighbors 
word = []
embeddings = []
labels = []
queries = []

#Selective Model for DMETM results
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=500, workers=4)
tsne_plot(model)

def nearest_neighbors(word, embeddings, vocab):
    # this function finds the 10 nearest neighbors for a given query word based on the Euclidean distance between words     in the embedding space. It takes the query word, embeddings of all words in the vocabulary and the vocabulary as arguemnts
    vectors = embeddings.copy()
    index = vocab.index(word)
    print('vectors: ', vectors.shape)
    query = vectors[index]
    print('query: ', query.shape)
    ranks = vectors.dot(query).squeeze()
    denom = query.T.dot(query).squeeze()
    denom = denom * np.sum(vectors**2, 1)
    denom = np.sqrt(denom)
    ranks = ranks / denom
    mostSimilar = []
    [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
    nearest_neighbors = mostSimilar[:10]
    nearest_neighbors = [vocab[comp] for comp in nearest_neighbors]
    return nearest_neighbors

def tsne_plot(embeddings, labels, queries, word="", colors="", names="", color_map="", color_flag=1, filename=""):
    print("Creates and TSNE model and plots it")
    if color_flag:
        custom_legend = []
        for name in names:
            custom_legend.append(Line2D([0], [0], color=color_map[name], lw=4))
    plt.figure(figsize=(20, 15))
    print("Model built...")
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2000, random_state=23)
    new_values = tsne_model.fit_transform(embeddings)
    print("Model fit...")
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 16
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        for i in range(len(x)):
            if color_flag:
                plt.scatter(x[i],y[i], c=colors[i])
                plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom',
                         color=colors[i])
                plt.legend(custom_legend, names, prop={'size': 30}, loc='center right')
            else:
                plt.scatter(x[i], y[i], c=colors[i])
                plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom',
                         c=colors[i])
                if labels[i] in queries:
                    #print(labels[i])
                    circ = plt.Circle((x[i], y[i]), 0.5, color='b', fill=False)
                    ax = plt.gca()
                    ax.add_artist(circ)
    plt.savefig(filename, bbox_inches='tight')
