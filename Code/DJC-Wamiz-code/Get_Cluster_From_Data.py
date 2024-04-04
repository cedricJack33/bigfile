import argparse
import pandas as pd
import html
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
# usage : python Get_Cluster_From_Data.py -p ../../Data/mini_new_messages.xlsx -l ""

emb_size = 768

def compute_features(new_table):


    clean_content = []
    for el in new_table['messages']:
        try:
            clean_content.append(html.unescape(el))
        except:
            clean_content.append('')

    new_table['clean_content'] = clean_content

    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    features = np.zeros((emb_size, len(new_table)))

    for ii,message in enumerate(new_table[:]['clean_content']):
        try:
            embeddings = model.encode([message])
            features[:,ii] = embeddings[0]
        except:
            pass

    return features


def compute_clusters(features, path_pca, path_kmeans):

    path = path_pca
    # open a file, where you stored the pickled data
    file = open(path, 'rb')
    # dump information to that file
    pca_ent = pickle.load(file)
    # close the file
    file.close()

    features_pca = pca_ent.transform(np.transpose(features))

    path = path_kmeans

    # open a file, where you stored the pickled data
    file = open(path, 'rb')

    # dump information to that file
    kmeans = pickle.load(file)

    # close the file
    file.close()

    dist_mots = distance_matrix(kmeans.cluster_centers_, features_pca)
    vec = np.argmin(dist_mots, axis=0)

    return(vec)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="args",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--path_new_messages", help="")
    parser.add_argument("-l", "--learn_str", help="")
    args = parser.parse_args()
    config = vars(args)

    path_new = config['path_new_messages']
    learn_str = config['learn_str']

    new_table = pd.read_excel(path_new)

    features = compute_features(new_table)
    path_kmeans = "../../Data/kmeans_" + learn_str + ".pickle"
    path_pca = "../../Data/pca_" + learn_str + ".pickle"

    vec = compute_clusters(features, path_pca, path_kmeans)
    new_table['clusters_found'] = vec
    new_table.to_excel("../../Data/new_messages_" + learn_str + ".xlsx")

    print("Done. File have been recorded in Data folder")
