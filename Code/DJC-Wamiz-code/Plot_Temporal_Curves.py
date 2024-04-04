import argparse
import pandas as pd
import html
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import datetime
from scipy import signal
import matplotlib
from matplotlib import pyplot as plt
# usage : python Plot_Temporal_Curves.py -c ../../Data/community_forum_category.csv -t ../../Data/community_forum_topic.csv -p ../../Data/community_forum_post.csv -n 259 -ys 2020 -pi ../../Data/images_test/ -pl ../../Data/CluFin.xlsx -l ""


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

def compute_images(labels, path_images, path_labels, year_start, data_cat):
    table_labels = pd.read_excel(path_labels)
    # print(table_labels)

    number_found = []
    for lab in range(200):

        vec = np.where(labels == lab)[0]

        for el in vec:

            str_date = data_cat.iloc[el]['created_at']
            year = int(str_date[:4])
            month = int(str_date[5:7])
            day = int(str_date[8:10])
            date_mess = datetime.datetime(year, month, day)
            date_beg = datetime.datetime(year_start, 1, 1)
            number = date_mess - date_beg
            number = number.days

            if number >= 0:
                number_found.append(number)

    max_all = np.max(number_found)
    min_all = np.min(number_found)


    window = signal.windows.gaussian(180, std=50)

    path = path_images

    temp_curves = []

    for lab in range(200):

        vec = np.where(labels == lab)[0]
        number_found = []
        for el in vec:

            str_date = data_cat.iloc[el]['created_at']
            year = int(str_date[:4])
            month = int(str_date[5:7])
            day = int(str_date[8:10])
            date_mess = datetime.datetime(year, month, day)
            date_beg = datetime.datetime(year_start, 1, 1)
            number = date_mess - date_beg
            number = number.days
            if number >= 0:
                number_found.append(number)

        pics = np.zeros(1+max_all-min_all)
        for el in number_found:
            pics[el-min_all] += 1
        temporal_curve = np.convolve(pics, window, mode='same')
        temporal_curve = temporal_curve[90:-90]
        temp_curves.append(temporal_curve)


    total_curve = np.zeros(len(temp_curves[0]))
    for el in temp_curves:
        total_curve = total_curve + el

    total_curve = total_curve/200
    plt.plot(total_curve)
    plt.savefig(path_images+'total' + '.png')
    plt.close()

    window = signal.windows.gaussian(180, std=50)

    temp_curves = []

    for lab in range(200):

        vec = np.where(labels == lab)[0]
        number_found = []
        for el in vec:

            str_date = data_cat.iloc[el]['created_at']
            year = int(str_date[:4])
            month = int(str_date[5:7])
            day = int(str_date[8:10])
            date_mess = datetime.datetime(year, month, day)
            date_beg = datetime.datetime(2020, 1, 1)
            number = date_mess - date_beg
            number = number.days
            if number >= 0:
                number_found.append(number)

        pics = np.zeros(1+max_all-min_all)
        for el in number_found:
            pics[el-min_all] += 1
        temporal_curve = np.convolve(pics, window, mode='same')
        temporal_curve = temporal_curve[90:-90]
        temp_curves.append(temporal_curve)

        plt.plot(np.divide(temporal_curve, total_curve))
        description = table_labels.iloc[np.where(table_labels['Cluster Number'] == lab)[0]]['focus']
        try:
            description = description.to_numpy()[0].replace('"', '').replace('?', '').replace('/', '')
        except:
            description = 'not a good cluster'
        plt.savefig(path_images+str(lab) + ' ' + description + '.png')
        plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="args",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--category", help="")
    parser.add_argument("-t", "--topic", help="")
    parser.add_argument("-p", "--post", help="")
    parser.add_argument("-n", "--num_cat", help="")
    parser.add_argument("-ys", "--year_start", help="")
    parser.add_argument("-pi", "--path_images", help="")
    parser.add_argument("-pl", "--path_labels", help="")
    parser.add_argument("-l", "--learn_str", help="")
    args = parser.parse_args()
    config = vars(args)
    path_category = config['category']
    path_topic = config['topic']
    path_post = config['post']
    num_cat = int(config['num_cat'])
    year_start = int(config['year_start'])
    path_images = config['path_images']
    path_labels = config['path_labels']
    learn_str = config['learn_str']


    path_kmeans = "../../Data/kmeans_" + learn_str + ".pickle"
    path_pca = "../../Data/pca_" + learn_str + ".pickle"
    path_features = "../../Data/features_" + learn_str + ".pickle"

    data_cat = create_table(path_category, path_topic, path_post)

    labels = compute_clusters(path_features, path_pca, path_kmeans)

    compute_images(labels, path_images, path_labels, year_start, data_cat)
