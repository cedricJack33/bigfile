import argparse
import pandas as pd
import html
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from transformers import pipeline
# usage : python Get_Emotion.py -p ../../Data/mini_new_messages.xlsx -l ""


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

    distilled_student_sentiment_classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        return_all_scores=True
    )

    clean_content = []
    for el in new_table['messages']:
        try:
            clean_content.append(html.unescape(el))
        except:
            clean_content.append('')

    new_table['clean_content'] = clean_content

    emotions = []
    levels = []
    yes_list = []
    for ii, message in enumerate(new_table['clean_content']):
        print(ii)

        try:
            message = message.replace('<p>', ' ').replace('</p>', ' ')
            res = distilled_student_sentiment_classifier(message)
            scores = []
            scores.append(res[0][0]['score'])
            scores.append(res[0][1]['score'])
            scores.append(res[0][2]['score'])

            list_prob = scores
            emotions.append(res[0][np.argmax(scores)]['label'])
            levels.append(scores[np.argmax(scores)])
            if (res[0][np.argmax(scores)]['label'] == 'positive' and scores[np.argmax(scores)] > 0.7) or (res[0][np.argmax(scores)]['label'] == 'negative' and scores[np.argmax(scores)] > 0.9):
                yes_list.append('yes')
            else:
                yes_list.append('no')

        except:
            emotions.append(-1)
            levels.append(-1)
            yes_list.append('no')



    new_table['emotions'] = emotions
    new_table['levels'] = levels
    new_table['yes_list'] = yes_list


    new_table.to_excel("../../Data/new_emo_messages_" + learn_str + ".xlsx")

    print("Done. File have been recorded in Data folder")
