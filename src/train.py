import time
import pickle
import os
import sys
import traceback
import string

import itertools

from github3.repos.repo import Repository
from github3.users import User

import nltk
from nltk.probability import FreqDist
from collections import Counter
from nltk.corpus import stopwords

from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

metaKeys = ["stargazers", "contributors", "subscribers", "forks", "teams"]
metaDictKeys = {"stargazers":"login", "contributors":"login", "subscribers":"login", "forks":"full_name", "teams":"full_name"}
DOWNLOAD_FOLDER = "dl"
FILES = ["README.md", "README", "readme.md", "readme"]
punctuation_translate_table = dict((ord(char), None) for char in string.punctuation)

DESC_ARRAY_SIZE = 256
README_ARRAY_SIZE = 512
USER_ARRAY_SIZE = 256

STARGAZER_WEIGHT, CONTRIBUTOR_WEIGHT, SUBSCRIBER_WEIGHT = 1, 2, 3

# Input
repos = []
repoMeta = []
# Processed
freqDescription = None
freqReadme = None
userIndexes = {}
repoStargazers, repoContributors, repoSubscribers = [], [], []
# Merged users
repoUsers = []

# ANN
inputData = None
model = None

def has_mask(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def has_mask_loss(y_true, y_pred):
    return -has_mask(y_true, y_pred)


def createKerasModel():
    inputs = Input((1, 100, 2000))

    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same')(x)

    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(1, 1, 1, activation='sigmoid')(x)

    model = Model(input=inputs, output=decoded)

    #model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=Adam(lr=1e-5), loss=has_mask_loss, metrics=[has_mask])


'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist

batch_size = 4
original_dim = DESC_ARRAY_SIZE + README_ARRAY_SIZE + USER_ARRAY_SIZE
latent_dim = 2
intermediate_dim = 128
epsilon_std = 0.01
nb_epoch = 10

def sampling(args):
    z_mean, z_log_std = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),
                              mean=0., std=epsilon_std)
    return z_mean + K.exp(z_log_std) * epsilon

class MyKerasModel:
    def vae_loss(self, x, x_decoded_mean):
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_std - K.square(self.z_mean) - K.exp(self.z_log_std), axis=-1) 
        return xent_loss + kl_loss

    def __init__(self):
        x = Input(batch_shape=(batch_size, original_dim))
        h = Dense(intermediate_dim, activation='relu')(x)
        self.z_mean = Dense(latent_dim)(h)
        self.z_log_std = Dense(latent_dim)(h)

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_std])`
        #z = Lambda(sampling, output_shape=(latent_dim,))([self.z_mean, self.z_log_std])
        z = Lambda(sampling)([self.z_mean, self.z_log_std])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(intermediate_dim, activation='relu')
        decoder_mean = Dense(original_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        self.vae = Model(x, x_decoded_mean)
        self.vae.compile(optimizer='rmsprop', loss=lambda x, x_decoded_mean: self.vae_loss(x, x_decoded_mean))

        # build a model to project inputs on the latent space
        self.encoder = Model(x, self.z_mean)

        # build a repository generator that can sample from the learned distribution
        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self.generator = Model(decoder_input, _x_decoded_mean)

def executeMNIST():
    model = MyKerasModel()

    # train the VAE on MNIST digits
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    model.vae.fit(x_train, x_train,
            shuffle=True,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            validation_data=(x_test, x_test))

    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = model.encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    plt.colorbar()
    plt.show()
    savefig("plot1.png")
    print("PLOT1")

    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # we will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-15, 15, n)
    grid_y = np.linspace(-15, 15, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]]) * epsilon_std
            x_decoded = model.generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure)
    plt.show()
    savefig("plot2.png")
    print("PLOT2")

#executeMNIST()

def loadRepos():
    index = 0
    while True:
        fileName = os.path.join(DOWNLOAD_FOLDER, 'repos' + str(index) + '.pkl')
        if not os.path.isfile(fileName):
            break
        print("Loading file: %s..." % fileName)
        for pickled in pickle.load(open(fileName, 'rb')):
            repo = Repository.from_dict(pickled["repo"])
            repos.append(repo)
            meta = pickled["meta"]
            repo.forks = [fork["full_name"] for fork in pickled["meta"]["forks"]]

            # reduce the meta dicts by only using the value of the metaDictKey
            for key in metaKeys:
                metaDictKey = metaDictKeys[key]
                valueList = meta.get(key) or []
                meta[key] = [value[metaDictKey] for value in valueList]
            meta["README"] = ""
            for FILE in FILES:
                if FILE in meta["files"]:
                    meta["README"] = meta["files"][FILE].decode(encoding='UTF-8')
                    meta["files"] = None
                    break

            repoMeta.append(meta)
            #for fork in repo.forks:
                #print(fork, pickled["repo"]["full_name"], Repository.from_dict(pickled["repo"]).full_name)

        index += 1
        #break
    return repos, repoMeta

def mergeRepos(repo1, meta1, repo2, meta2):
    for key in metaKeys:
        meta1[key] = set(meta1[key]).union(set(meta2[key]))
    if meta1["README"] == "":
        meta1["README"] = meta2["README"]

def forkMerge():
    global repos, repoMeta
    forkNameIndex = {}
    synsets = []
    print("Fork merge...")
    print("*"*30)
    print("Preparing synonym sets...")
    for i, repo in enumerate(repos):
        forkNameIndex[repo.full_name] = i
        synsets.append([])
    for i, repo in enumerate(repos):
        for fork in repo.forks:
            index = forkNameIndex.get(fork)
            if index is not None:
                synsets[i].append(index)
                synsets[index].append(i)
    print("Merging synonym sets...")
    mergedRepos = []
    mergedRepoMeta = []
    for i, synset in enumerate(synsets):
        # check if already merged
        if len(synsets[i]) != 0 and min(synsets[i]) < i:
            continue
        repo1, meta1 = repos[i], repoMeta[i]
        for index in synset:
            repo2, meta2 = repos[index], repoMeta[i]
            mergeRepos(repo1, meta1, repo2, meta2)
        mergedRepos.append(repo1)
        mergedRepoMeta.append(meta1)
    print("Repositories before merge: %d" % len(repos))
    print("Repositories after merge: %d" % len(mergedRepos))
    repos = mergedRepos
    repoMeta = mergedRepoMeta
    # cache to file
    output = {"repos":[{"full_name":repo.full_name, "description":repo.description} for repo in repos], "repoMeta":repoMeta}
    pickle.dump(output, open("merged_repos.pkl", "wb"))

def preprocess():
    loadRepos()

def processText(text):
    lowers = text.lower()
    no_punctuation = lowers.translate(punctuation_translate_table)
    tokens = nltk.word_tokenize(no_punctuation)
    filtered = [w for w in tokens if not w in stopwords.words('english')]
    return filtered

def tokens2Array(tokens, arraySize, key, startIndex):
    counter = Counter(tokens)
    common = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
    common = common[:arraySize]

    commonKeys = {}
    for i, item in enumerate(common):
        commonKeys[item[0]] = i

    for i, repo in enumerate(repos):
        if key == "users":
            for user in repoStargazers[i]:
                tokenIndex = commonKeys.get(user)
                if tokenIndex:
                    inputData[i, tokenIndex+startIndex] += STARGAZER_WEIGHT
            for user in repoContributors[i]:
                tokenIndex = commonKeys.get(user)
                if tokenIndex:
                    inputData[i, tokenIndex+startIndex] += CONTRIBUTOR_WEIGHT
            for user in repoSubscribers[i]:
                tokenIndex = commonKeys.get(user)
                if tokenIndex:
                    inputData[i, tokenIndex+startIndex] += SUBSCRIBER_WEIGHT
        else:
            if key == "description": #HACK
                text = repo.get(key)
            else:
                text = repoMeta[i].get(key)
            if text is not None and text != "":
                tokens = processText(text)
                tokens = list(set(tokens))
                for token in tokens:
                    tokenIndex = commonKeys.get(token)
                    if tokenIndex:
                        inputData[i, tokenIndex+startIndex] = 1

    return common, commonKeys

def processDescription():
    global freqDescription
    allTextTokens = []
    hasDesc = 0
    for repo in repos:
        if repo["description"] is not None:
            if repo["description"] != "":
                hasDesc += 1
            tokens = processText(repo["description"])
            tokens = list(set(tokens))
            allTextTokens += tokens
    print("Total description files: %d, in %.2f%% repositories" % (hasDesc, 100*hasDesc / len(repos)))
    return tokens2Array(allTextTokens, DESC_ARRAY_SIZE, "description", 0)

def processReadmes():
    global freqReadme
    allTextTokens = []
    hasRepo = 0
    for meta in repoMeta:
        if meta["README"] != "":
            hasRepo += 1
        tokens = processText(meta["README"])
        tokens = list(set(tokens))
        allTextTokens += tokens
    print("Total readme files: %d, in %.2f%% repositories" % (hasRepo, 100*hasRepo / len(repoMeta)))
    return tokens2Array(allTextTokens, README_ARRAY_SIZE, "README", DESC_ARRAY_SIZE)

def addUser(repoIndx, user, mapping):
    if not user in userIndexes:
        userIndx = len(userIndexes) + 1
        userIndexes[user] = userIndx
        mapping[repoIndx].append(userIndx)

        #for v in maps.values():
        #mapping[username] = 1
    #else:
        #mapping[username] += 1

def processUsers():
    print("Processing users...")
    print("*"*30)

    users = []
    for i, meta in enumerate(repoMeta):
        #repoUsers.append([])
        repoStargazers.append([])
        repoContributors.append([])
        repoSubscribers.append([])
        for user in meta["stargazers"]:
            addUser(i, user, repoStargazers)
        for user in meta["contributors"]:
            addUser(i, user, repoContributors)
        for user in meta["subscribers"]:
            addUser(i, user, repoSubscribers)
        users += list(set(repoStargazers[i] + repoContributors[i] + repoSubscribers[i]))
        sys.stdout.write("\rProcessed %d/%d repositories." % (i, len(repoMeta)))
    print()
    print("Total users: %d" % len(userIndexes))
    allSubscribers = list(itertools.chain.from_iterable(repoSubscribers))
    allStargazers = list(itertools.chain.from_iterable(repoStargazers))
    allContributors = list(itertools.chain.from_iterable(repoContributors))

    print("Stargazing: %d, contributing: %d, subscribing: %d" % (len(allStargazers),len(allContributors), len(allSubscribers)))

    print("Repo average Stargazing: %.4f, contributing: %.4f, subscribing: %.4f" % (len(allStargazers)/len(repos),len(allContributors)/len(repos), len(allSubscribers)/len(repos)))

    return tokens2Array(users, USER_ARRAY_SIZE, "users", DESC_ARRAY_SIZE + README_ARRAY_SIZE)
    #pickle.dump({"userIndexes":userIndexes, "repoStargazers":repoStargazers, "repoContributors":repoContributors, "repoSubscribers":repoSubscribers}, open("user_indexes.pkl", 'wb'))
    #
    #sum(usersToOccurance.values())/len(usersToOccurance))

def plot():
    import matplotlib.cm as cmx
import matplotlib.colors as clrs

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = clrs.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color


    import matplotlib.patches as mpatches

    #print(readme)
    #readmeKeys['ruby']
    #readmeKeys['rails']
    #readmeKeys['gem']
    #readmeKeys['javascript']
    #readmeKeys['c']
    #readmeKeys['python']
    colorKeys = {}
    colorKeys['ruby'] = 1
    colorKeys['rails'] = 1
    colorKeys['gem'] = 1
    colorKeys['python'] = 2
    colorKeys['django'] = 2
    colorKeys['javascript'] = 3
    colorKeys['jquery'] = 3
    colorKeys['php'] = 4
    colorKeys['perl'] = 5
    colorKeys['c'] = 6 
    colorKeys['html'] = 7
    colorKeys['erlang'] = 8


    colors = np.zeros(inputData.shape[0])
    for i, data in enumerate(inputData):
        for colorKey in colorKeys.keys():
            index = descriptionKeys[colorKey]
            if inputData[i, index] == 1:
                colors[i] = colorKeys[colorKey]
                break
    print()
    plotData = inputData[colors!=0]
    colors = colors[colors!=0]
    colors = cmap(colors)

    colorKeys['rails'] = None
    colorKeys['gem'] = None
    colorKeys['django'] = None
    colorKeys['jquery'] = None
    invColorKeys = {v: k for k, v in colorKeys.items()}
    labels = [invColorKeys[i] for i in range(1, 9)]
    cmap = get_cmap(len(invColorKeys))

    # display a 2D plot of the repository classes in the latent space
    #x_test_encoded = model.encoder.predict(inputData, batch_size=batch_size)
    x_test_encoded = model.encoder.predict(plotData, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=colors)

    recs = []
    for i in range(1,len(invColorKeys)+1):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=cmap(i)))
    plt.legend(recs,labels,loc=4)

    #plt.legend(scatter,
    #           [invColorKeys[i] for i in range(1, 9)],
    #           scatterpoints=1,
    #           loc='lower left',
    #           ncol=3,
    #           fontsize=8)

    plt.show()
    savefig("plot1-repo.png")
    print("Plotted repository map")


def execute():
    global repos, repoMeta, inputData, model
    print("Loading repos...")
    print("*"*30)
    try:
        print("Loading preprocessed merged repositories...")
        print("*"*30)
        pickled = pickle.load(open("merged_repos.pkl", "rb"))
        repos, repoMeta = pickled["repos"], pickled["repoMeta"]
    except Exception as x:
        print(x)
        traceback.print_exc()
        preprocess()
        forkMerge()
    print("Total repositories loaded: %d" % len(repos))

    try:
        print("Loading preprocessed input data...")
        print("*"*30)
        inputData = pickle.load(open("input_data_nn.pkl", "rb"))
        #print("Loading preprocessed user indexes repositories...")
        #print("*"*30)
        #pickled = pickle.load(open("user_indexes.pkl", 'rb'))
        #userIndexes = pickled["userIndexes"]
        #repoUsers = pickled["repoUsers"]

        #print(userIndexes, repoUsers)
    except Exception as x:
        print(x)
        traceback.print_exc()

        inputData = np.zeros((len(repos), original_dim))
        users, usersKeys = processUsers()
        description, descriptionKeys = processDescription()
        readme, readmeKeys = processReadmes()
        pickle.dump({"description":description, "descriptionKeys":descriptionKeys, "readme":readme, "readmeKeys":readmeKeys, "users":users, "usersKeys":usersKeys}, 
                    open("input_data_processed.pkl", "wb"))

        pickle.dump(inputData, open("input_data_nn.pkl", "wb"))

    model = MyKerasModel()
    model.vae.fit(inputData, inputData,
            shuffle=True,
            nb_epoch=nb_epoch,
            batch_size=batch_size,
            validation_data=(inputData, inputData))

execute()
