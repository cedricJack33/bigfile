import argparse
import pandas as pd
import html
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# usage : python Look_At_Messages.py -c ../../Data/community_forum_category.csv -t ../../Data/community_forum_topic.csv -p ../../Data/community_forum_post.csv -n 259 -cn 113 -l ""

emb_size = 768

def create_table(path_category, path_topic, path_post):

    # Extract Minimal DB Needed
    data_table_category = pd.read_csv(path_category, sep=';')

    id_np = data_table_category['id'].to_list()
    name_np = data_table_category['name'].to_list()

    data_table_category_s = pd.DataFrame()
    data_table_category_s['category_id'] = id_np
    data_table_category_s['category_name'] = name_np

    data_table_topic = pd.read_csv(path_topic, sep=';')

    id_np = data_table_topic['id'].to_list()
    title_np = data_table_topic['title'].to_list()
    category_id_np = data_table_topic['category_id'].to_list()

    data_table_topic_s = pd.DataFrame()
    data_table_topic_s['topic_id'] = id_np
    data_table_topic_s['topic_title'] = title_np
    data_table_topic_s['category_id'] = category_id_np

    data_table_topic_category_s = pd.merge(data_table_topic_s, data_table_category_s, on=["category_id"])

    data_table_post = pd.read_csv(path_post, sep=';')

    uuid_np = data_table_post['uuid'].to_list()
    topic_id_np = data_table_post['topic_id'].to_list()
    raw_content_np = data_table_post['raw_content'].to_list()
    content_np = data_table_post['content'].to_list()
    created_at_np = data_table_post['created_at'].to_list()
    author_id_np = data_table_post['author_id'].to_list()

    data_table_post_s = pd.DataFrame()
    data_table_post_s['post_uuid'] = uuid_np
    data_table_post_s['content'] = content_np
    data_table_post_s['raw_content'] = raw_content_np
    data_table_post_s['created_at'] = created_at_np
    data_table_post_s['author_id'] = author_id_np
    data_table_post_s['topic_id'] = topic_id_np

    data_table_post_topic_category_s = pd.merge(data_table_post_s, data_table_topic_category_s, on=["topic_id"])

    data_cat = data_table_post_topic_category_s[data_table_post_topic_category_s['category_id'] == 259]
    clean_content = []
    for el in data_cat['raw_content']:
        try:
            clean_content.append(html.unescape(el))
        except:
            clean_content.append('')

    data_cat['clean_content'] = clean_content
    return data_cat

def compute_clusters(path_features, path_pca, path_kmeans):
    import pickle
    path = path_features
    # open a file, where you stored the pickled data
    file = open(path, 'rb')
    # dump information to that file
    features = pickle.load(file)
    # close the file
    file.close()

    path = path_pca
    # open a file, where you stored the pickled data
    file = open(path, 'rb')
    # dump information to that file
    pca_ent = pickle.load(file)
    # close the file
    file.close()

    features_pca = pca_ent.transform(np.transpose(features))

    import pickle

    path = path_kmeans

    # open a file, where you stored the pickled data
    file = open(path, 'rb')

    # dump information to that file
    kmeans = pickle.load(file)

    # close the file
    file.close()

    return kmeans.labels_



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="args",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--category", help="")
    parser.add_argument("-t", "--topic", help="")
    parser.add_argument("-p", "--post", help="")
    parser.add_argument("-n", "--num_cat", help="")
    parser.add_argument("-cn", "--clust_nb", help="")
    parser.add_argument("-l", "--learn_str", help="")
    args = parser.parse_args()
    config = vars(args)
    path_category = config['category']
    path_topic = config['topic']
    path_post = config['post']
    num_cat = int(config['num_cat'])
    cluster_number = int(config['clust_nb'])
    learn_str = config['learn_str']


    path_kmeans = "../../Data/kmeans_" + learn_str + ".pickle"
    path_pca = "../../Data/pca_" + learn_str + ".pickle"
    path_features = "../../Data/features_" + learn_str + ".pickle"

    data_cat = create_table(path_category, path_topic, path_post)

    labels = compute_clusters(path_features, path_pca, path_kmeans)

    vec = np.where(labels == cluster_number)[0]
    end = np.min([15, len(vec)])
    for ii, el in enumerate(vec[:end]):
        print(ii)

        print(data_cat.iloc[el]['clean_content'])
        print('\n')
