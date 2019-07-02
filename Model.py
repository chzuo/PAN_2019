#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys, getopt, json, os, re, random, pickle
import keras
import tensorflow
import scipy
from nltk.tokenize import word_tokenize, sent_tokenize
from scipy.cluster.hierarchy import linkage, inconsistent, fcluster
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk import pos_tag
import textstat as ts
from numpy import asarray
from numpy import max as npmax
import numpy.core._methods
import numpy.lib.format
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.externals import joblib

train_folder = 'data/Training'
vectorizer = joblib.load('vectorizer_model.sav')
similarity_model = joblib.load('similarity_model.sav')
hcluster_model = pickle.load(open('mlp_4_cluster', 'rb'))
bicla_model = pickle.load(open('base_mlp_model', 'rb'))


def optiKMean(X):
    try:
        up_bound = min(6,len(X-1))
        result = 2
        score = -1000000000000000
        for k in range(2, up_bound):
            estimator = KMeans(n_clusters=k)
            estimator.fit(X)
            s = silhouette_score(X, estimator.labels_, metric='euclidean')
            if s > score:
                score = s
                result = k
        return result
    except Exception:
        return result


def tfidf_optiKMean_predict(X):
    return [optiKMean(vectorizer.transform(x).toarray()) for x in X]


def pred(clf, X):
    return clf.predict_proba(X.reshape(1,-1))[0][1]


def cor_matrix(X):
    size = len(X)
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i][j] = 1
            if i > j:
                v = pred(similarity_model, np.hstack([X[i], X[j]]))
                matrix[i][j] = v
                matrix[j][i] = v
    return matrix

def similarity_optiKMean_predict(X):
    return [optiKMean(cor_matrix(vectorizer.transform(x).toarray())) for x in X]


def write_truth(filename, num_author):
    dict = {"authors": int(num_author)}
    with open(filename, 'w') as f:
        json.dump(dict, f, indent=4)


def write_folder(foldername, index_list, truth_list):
    for index,truth in zip(index_list,truth_list):
        truth_file = foldername + '/problem-' + str(index) + '.truth'
        truth = int(truth)
        write_truth(truth_file, truth)

def get_raw_data(foldername, train_flag):
    if train_flag:
        num_file = int((len(os.listdir(foldername)) ) / 2)
    else:
        num_file = int(len(os.listdir(foldername)))

    print(num_file)
    text_list, tag_list, index_list = [],[],[]

    for index in range(1, num_file + 1):
        index_list.append(index)
        txt_filename = foldername + '/problem-' + str(index) + '.txt'

        with open(txt_filename, 'r', encoding='utf-8') as f:
            text = ' '.join(line for line in f.readlines())
            text_list.append(text)

        if train_flag:
            truth_filename = foldername + '/problem-' + str(index) + '.truth'
            with open(truth_filename, 'r', encoding='utf-8') as rf:
                truth = json.load(rf)
                num_authors = 2 if truth['authors']>1 else 1
                tag_list.append(num_authors)

    return text_list, tag_list, index_list

def get_raw_data_v4(foldername, index_list, train_flag):

    text_list, tag_list, new_index_list = [],[],[]

    for index in index_list:

        txt_filename = foldername + '/problem-' + str(index) + '.txt'

        with open(txt_filename, 'r', encoding='utf-8') as f:
            text = ' '.join(line for line in f.readlines())

        if train_flag:
            truth_filename = foldername + '/problem-' + str(index) + '.truth'
            with open(truth_filename, 'r', encoding='utf-8') as rf:
                truth = json.load(rf)
                num_authors = truth['authors']
                if num_authors >1:
                    tag_list.append(num_authors)
                    new_index_list.append(index)
                    text_list.append(text)

        else:
            new_index_list.append(index)
            text_list.append(text)

    return text_list, tag_list, new_index_list

def get_cluster_data(input_folder, output_folder, index_list, train_flag):
    text_list = []
    tag_dict = {}
    new_index_list = []
    tag_list = []

    if train_flag:
        train_index_list = []
        for index in index_list:
            truth_filename = input_folder + '/problem-' + str(index) + '.truth'
            with open(truth_filename, 'r', encoding='utf-8') as rf:
                truth = json.load(rf)
                num_authors = truth['authors']
                if num_authors > 1:
                    tag_dict[index] = num_authors
                    train_index_list.append(index)
        index_list = train_index_list

    for index in index_list:
        filename = input_folder+ '/problem-'+str(index)+'.txt'

        with open(filename, 'r', encoding='utf-8') as f:
            text = ''
            for line in f.readlines():
                if line.startswith('Terminal') or line.startswith('Try it online') or line.startswith('$URL$') or \
                        line.startswith('OSX 10.11.2'):
                    continue
                text += line
            new_text = text.replace('$URL$', '')
            new_text = new_text.replace(' \n', '\n')

        new_text = re.sub(r'[\n]{3,}', '\n\n', text)
        all_paras = [t for t in new_text.split('\n\n') if len(t) > 30]
        all_newline = [t for t in new_text.split('\n') if len(t) > 30]
        #lang = detect(text)


        #if not English, or num_newlins <5
        if len(all_newline) < 5: #lang != 'en' or
            truth_file = output_folder + '/problem-'+str(index)+'.truth'
            write_truth(truth_file, 1)
            continue

        if len(all_newline)<12:
            truth_file = output_folder + '/problem-'+str(index)+'.truth'
            write_truth(truth_file, random.randint(1,4))
            continue

        if len(all_paras) > 15:  # use paras for clustering
            text_split = all_paras
        else:
            text_split = all_newline

        text_list.append(text_split)
        new_index_list.append(index)

    if train_flag:
        for index in new_index_list:
            tag_list.append(tag_dict[index])

    return text_list, tag_list, new_index_list


def count_occurence(check_word_list, word_list_all):
    num_count = 0
    for w in check_word_list:
        if w in word_list_all:
            num_count += word_list_all[w]
    return num_count

def count_occurence_phrase(phrase_list, paras):
    num_count = 0
    for phrase in phrase_list:
        num_count += paras.count(phrase)
    return num_count


def extract_feature_file(text):
    feature_all = []
    for paras in text:

        sent_list = sent_tokenize(paras)
        word_dict = {}

        sent_length_list = [0,0,0,0,0,0] #0-10,10-20,20-30,30-40,40-50,>50
        pos_tag_list =[ 0 for i in range(15)]
        for sent in sent_list:
            w_list = word_tokenize(sent)
            for (word, tag) in pos_tag(w_list):
                if tag in ['PRP']:
                    pos_tag_list[0] += 1
                if tag.startswith('J'):
                    pos_tag_list[1] += 1
                if tag.startswith('N'):
                    pos_tag_list[2] += 1
                if tag.startswith('V'):
                    pos_tag_list[3] += 1
                if tag in ['PRP', 'PRP$', 'WP', 'WP$']:
                    pos_tag_list[4] += 1
                elif tag in ['IN']:
                    pos_tag_list[5] += 1
                elif tag in ['CC']:
                    pos_tag_list[6] += 1
                elif tag in ['RB', 'RBR', 'RBS']:
                    pos_tag_list[7] += 1
                elif tag in ['DT', 'PDT', 'WDT']:
                    pos_tag_list[8] += 1
                elif tag in ['UH']:
                    pos_tag_list[9] += 1
                elif tag in ['MD']:
                    pos_tag_list[10] += 1
                if len(word) >= 8:
                    pos_tag_list[11] += 1
                elif len(word) in [2, 3, 4]:
                    pos_tag_list[12] += 1
                if word.isupper():
                    pos_tag_list[13] += 1
                elif word[0].isupper():
                    pos_tag_list[14] += 1

            num_words_sent = len(w_list)
            if num_words_sent >= 50:
                sent_length_list[-1] += 1
            else:
                sent_length_list[int(num_words_sent/10)] += 1

            for w in w_list :
                if len(w)>20:
                    w = '<Long_word>'
                word_dict.setdefault(w, 0)
                word_dict[w] += 1
        base_feat1 = [len(sent_list), len(word_dict)] + sent_length_list + pos_tag_list #num_sentences, num_words

        #print('base_feat', base_feat1)

        special_char = [';',':','(','/','&',')','\\','\'','"','%','?','!','.','*','@']
        char_feat = [paras.count(char) for char in special_char]

        with open('function_words.json', 'r') as f:
            function_words = json.load(f)

        function_words_feature = []
        for w in function_words['words']:
            if w in word_dict:
                function_words_feature.append(word_dict[w])
            else:
                function_words_feature.append(0)

        function_phrase_feature = [paras.count(p) for p in function_words['phrases']]

        with open('difference_word.json', 'r') as f:
            difference_dict = json.load(f)

        difference_words_feat = []
        difference_words_feat.append(count_occurence(difference_dict['word']['number'][0], word_dict))
        difference_words_feat.append(count_occurence(difference_dict['word']['number'][1], word_dict))
        difference_words_feat.append(count_occurence(difference_dict['word']['spelling'][0], word_dict))
        difference_words_feat.append(count_occurence(difference_dict['word']['spelling'][1], word_dict))
        difference_words_feat.append(count_occurence_phrase(difference_dict['phrase'][0], paras))
        difference_words_feat.append(count_occurence_phrase(difference_dict['phrase'][1], paras))

        textstat_feat = [ts.flesch_reading_ease(paras), ts.smog_index(paras),ts.flesch_kincaid_grade(paras), \
                         ts.coleman_liau_index(paras), ts.automated_readability_index(paras),\
                         ts.dale_chall_readability_score(paras), ts.difficult_words(paras), ts.linsear_write_formula(paras),\
                         ts.gunning_fog(paras)]

        feature = base_feat1 + function_words_feature + function_phrase_feature + difference_words_feat + char_feat + textstat_feat
        feature_all.append(feature)

        #print(feature)

    return asarray(feature_all)


def clustering_train(text_list, tag_list, index_list):

    for text, tag, index in zip(text_list, tag_list, index_list):
        feature_all = extract_feature_file(text)
        #print(feature_all.shape)

        Z = linkage(feature_all, 'ward')

        Y = inconsistent(Z)
        #c, coph_dists = cophenet(Z, pdist(feature_all))

        distance_last_5 = Y[-5:, 3].tolist()

        distance_last_5.append(800)
        d1 = distance_last_5[5-tag]
        d2 = distance_last_5[5-tag+1]
        #print(d1,d2,':', distance_last_5, tag)

        distance_last_5 = Z[-5:, 2].tolist()
        distance_last_5.append(800)
        d1 = distance_last_5[5-tag]
        d2 = distance_last_5[5-tag+1]
        #print(d1,d2,':', distance_last_5, tag)

    cutoff  = 1
    return cutoff

def class_step2_trains(text_list, tag_list, index_list):
    for i in range(len(tag_list)):
        tag = tag_list[i]
        tag_list[i] = tag - 2
    #print(tag_list)
    feat_all = []
    for text, tag, index in zip(text_list, tag_list, index_list):

        feature_all = extract_feature_file(text)
        #print(feature_all.shape)

        Z = linkage(feature_all, 'ward')

        Y = inconsistent(Z)
        #c, coph_dists = cophenet(Z, pdist(feature_all))

        distance_last_10 = Z[-30:, 2].tolist()
        incon_last_10 = Y[-30:, 3].tolist()
        feat = distance_last_10 + incon_last_10
        feat_all.append(feat)

    x_train = pad_sequences(feat_all, maxlen=60, dtype='float32', padding='pre', truncating='pre',value=0.0)
    #x_train = np.asarray(feat_all, dtype=np.float32)
    y_train = to_categorical(tag_list, num_classes=4)

    model = Sequential()
    model.add(Dense(10, input_shape=(60,), activation='relu'))
    #model.add(Dense(6, activation='sigmoid'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, validation_split=0)
    #model_file = 'mlp_4_cluster'
    #pickle.dump(model, open(model_file, 'wb'))

    #model = pickle.load(open(model_file, 'rb'))

    return model


def class_step2_test(text_list):
    feat_all = []
    for text in text_list:
        feature_all = extract_feature_file(text)
        #print(feature_all.shape)

        Z = linkage(feature_all, 'ward')

        Y = inconsistent(Z)
        # c, coph_dists = cophenet(Z, pdist(feature_all))

        distance_last_10 = Z[-30:, 2].tolist()
        #print(distance_last_10)
        incon_last_10 = Y[-30:, 3].tolist()
        #print(incon_last_10)
        feat = distance_last_10 + incon_last_10
        feat_all.append(feat)

    #x_test= np.asarray(feat_all)
    x_test = pad_sequences(feat_all, maxlen=60, dtype='float32', padding='pre',
                                                         truncating='pre', value=0.0)

    predictions = hcluster_model.predict(x_test, batch_size=32, verbose=0)
    predictions = predictions.argmax(axis=-1)
    for i in range(len(predictions)):
        predictions[i] = int(predictions[1])+2

    return predictions


def clustering_test(text_list, cutoff, index_list, outfut_folder):
    truth_list = []

    for text, index in zip(text_list, index_list):
        feature_all = extract_feature_file(text)
        #print(feature_all.shape)

        Z = linkage(feature_all, 'ward')
        #c, coph_dists = cophenet(Z, pdist(feature_all))
        clusters = fcluster(Z, cutoff, criterion='inconsistent')
        num_authors = npmax(clusters)
        num_authors = min(num_authors,5)
        num_authors = max(num_authors,2)
        truth_list.append(int(num_authors))

    write_folder(outfut_folder, index_list, truth_list)


def main(argv):
    input_folder = ''
    output_folder = ''

    try:
         opts, args = getopt.getopt(argv,"hi:o:",["ifolder=","ofolder="])
    except getopt.GetoptError:
         print('test.py -i <input_folder> -o <output_folder>')
         sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <input_folder> -o <output_folder>')
            sys.exit()
        elif opt in ("-i", "--ifolder"):
            input_folder = arg
        elif opt in ("-o", "--ofolder"):
            output_folder = arg
    print('Input folder is ', input_folder)
    print('Output folder is ', output_folder)

    if input_folder.endswith('/'):
        input_folder = input_folder[:-1]
    if output_folder.endswith('/'):
        output_folder = output_folder[:-1]

    try:
        os.mkdir(output_folder)
    except:
        print("Folder exists:",output_folder)


    ######################################################################################################
    # Step 1: Binary classification
    ######################################################################################################


    x_train, y_train, index_list_train = get_raw_data(train_folder, train_flag=True)
    x_test, y_test, index_list_test = get_raw_data(input_folder,train_flag=False)

    all_text = x_train + x_test

    print("Training on {0} examples".format(len(x_train)))

    tokenizer = Tokenizer(num_words=40000, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                                   lower=True, split=" ", char_level=False, oov_token=None)

    tokenizer.fit_on_texts(all_text)

    x_train = tokenizer.texts_to_matrix(x_train, mode='tfidf')
    x_test = tokenizer.texts_to_matrix(x_test, mode='tfidf')

    num_class = npmax(y_train)+1
    y_train = to_categorical(y_train, num_classes=num_class)
    '''
    model_file = 'base_mlp_model'

    model = Sequential()
    model.add(Dense(128, input_shape=(40000,), activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print('Start train MLP model')
    history = model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=1, validation_split=0.1)
    '''

    predictions = bicla_model.predict(x_test, batch_size=32, verbose=0)
    predictions = predictions.argmax(axis=-1)

    p1_p, index_p1= [], []  #one-author file
    p2_p, index_p2 = [], [] #multi_author file
    for prep, index in zip(predictions, index_list_test):
        if prep == 1:
            p1_p.append(int(prep))
            index_p1.append(index)
        else:
            p2_p.append(int(prep))
            index_p2.append(index)

    write_folder(output_folder, index_p1, p1_p)

    print('Step one Done')

    ######################################################################################################
    # Step 2: Author clustering
    ######################################################################################################
    '''
    x_train_clu, y_train_clu, index_train_clu = get_cluster_data(train_folder, output_folder , index_list_train, train_flag=True)
    x_test_clu, _, index_test_clu = get_cluster_data(input_folder, output_folder, index_p2, train_flag=True)

    model = class_step2_trains(x_train_clu, y_train_clu, index_train_clu)
    class_step2_test(x_test_clu, index_test_clu, model, output_folder)
    '''
    x_test_clu, _, index_test_clu = get_cluster_data(input_folder, output_folder, index_p2, train_flag=False)

    prediction_kmeans = tfidf_optiKMean_predict(x_test_clu)
    print('Kmeans Done')
    prediction_simi = similarity_optiKMean_predict(x_test_clu)
    print('Simi Done')
    predictions_hclus = class_step2_test(x_test_clu)
    print('hclus Done')

    final_prep = []
    for k, s, h in zip(prediction_kmeans, prediction_simi,predictions_hclus):
        prep = int((k+s+h)/3)
        final_prep.append(prep)

    write_folder(output_folder, index_test_clu, final_prep)


if __name__ == '__main__':
    main(sys.argv[1:])
