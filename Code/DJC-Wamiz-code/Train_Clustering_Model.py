import argparse
import pandas as pd
import html
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# usage : python Train_Clustering_Model.py -c ../../Data/community_forum_category.csv -t ../../Data/community_forum_topic.csv -p ../../Data/community_forum_post.csv -n 259 -l "on_part_1"

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

def compute_features(data_cat):
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    features = np.zeros((emb_size, len(data_cat)))

    for ii,message in enumerate(data_cat[:]['clean_content']):
        try:
            embeddings = model.encode([message])
            features[:,ii] = embeddings[0]
        except:
            pass

    path = "../../Data/features_" + learn_str +".pickle"
    filehandler = open(path, 'wb')
    pickle.dump(features, filehandler)
    filehandler.close()

    return features

def comupte_pca(features):
    pca = PCA(n_components=60)
    pca_ent = pca.fit(np.transpose(features))

    path = "../../Data/pca_" + learn_str + ".pickle"
    filehandler = open(path, 'wb')
    pickle.dump(pca_ent, filehandler)
    filehandler.close()

    return pca

def compute_clusters(features, pca):
    features_pca = pca.transform(np.transpose(features))
    kmeans = KMeans(n_clusters=200, random_state=0, n_init="auto").fit(features_pca)

    path = "../../Data/kmeans_" + learn_str + ".pickle"
    filehandler = open(path, 'wb')
    pickle.dump(kmeans, filehandler)
    filehandler.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="args",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--category", help="")
    parser.add_argument("-t", "--topic", help="")
    parser.add_argument("-p", "--post", help="")
    parser.add_argument("-n", "--num_cat", help="")
    parser.add_argument("-l", "--learn_str", help="")
    args = parser.parse_args()
    config = vars(args)
    path_category = config['category']
    path_topic = config['topic']
    path_post = config['post']
    num_cat = int(config['num_cat'])
    learn_str = config['learn_str']

    data_cat = create_table(path_category, path_topic, path_post)
    data_cat = data_cat[:200] # to remove

    features = compute_features(data_cat)
    pca = compute_pca(features)
    print(pca)
    compute_clusters(features, pca)
    print("Done. Files have been recorded in Data folder")
