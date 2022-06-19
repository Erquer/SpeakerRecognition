import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from Speaker import Speaker
from knn import Knn

SAMPLING_RATE = 16000
hop_len = 512
prefixes = ['BJ', 'HN', 'PK', 'PT', 'ZZ']
test_mfcc = [-388.3489471435547, 13.547713041305542, -5.369972674548626, 58.523172760009764, -17.565272903442384, 13.892732334136962, -5.140625596046448, 6.523916959762573, 1.3403185039758683, 3.3588162899017333, 8.3439462184906, 1.376888209581375, 4.443789267539978]
# test_mfcc = [-388.3489471435547, 49.38297576904297, 7.8444747686386105, 41.1297794342041, -0.16219628229737282, -0.0397356778383255, -1.5765647903084754, 4.141047525405884, 2.1754455413669347, 5.320105957984924, 2.952728569507599, 5.791269254684448, 1.2452065262012184]
# test_mfcc = [-195.9180160522461, 16.14417428970337, 4.962639139592648, 58.52317314147949, -25.354970359802245, 23.945265197753905, -6.618231773376465, -8.897641658782959, 6.040571284294129, 7.052741050720215, 2.785193556547165, 2.3840505599975588, 4.443789267539978]
def get_mfcc(prefix, dict):
    for i in range(1, 11):
        filename = f'./dataset/pocięte/{prefix}_{i}.wav'
        y, sr = librosa.load(filename, sr=SAMPLING_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        dict[f'{prefix}_{i}'] = mfcc

def get_mfcc_mean(prefix):
    mfccs = []
    for i in range(1, 11):
        filename = f'./dataset/pocięte/{prefix}_{i}.wav'
        y, sr = librosa.load(filename, sr=SAMPLING_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mean_mfcc = mfcc.mean(axis=1)
        mfccs.append(mean_mfcc)
    return mfccs

def get_all_mfcc_means(dict):
    mfcc_means = []
    speakers = []
    i = 0
    for prefix in prefixes:
        # oblicz listę współczynników jednej osoby dla wszyskich nagrań
        mfcc_table = np.array(get_mfcc_mean(prefix)).T.tolist()
        # dodaj średnią z każdego współczynnika do tablicy średnich
        speakers.append(Speaker([np.mean(table) for table in mfcc_table], prefix))
        mfcc_means.append([np.mean(table) for table in mfcc_table])
        # print(f'[{prefix}] mfcc_means: {mfcc_means[i]}')
        # Dodaj do słownika średnie
        dict[prefix] = mfcc_means[i]
        i += 1
    return [mfcc_means, speakers]
def get_all_mfcc(dict):
    for prefix in prefixes:
        get_mfcc(prefix, dict)

def get_all_speakers():
    speakers = []
    for prefix in prefixes:
        for i in range(1, 11):
            filename = f'./dataset/pocięte/{prefix}_{i}.wav'
            y, sr = librosa.load(filename, sr=SAMPLING_RATE)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # zależne od dł
            # for record in mfcc.T.tolist():
            #     speakers.append(Speaker(record, prefix))
            table = np.array(mfcc.T.tolist())
            mean = table.mean(axis=0)
            speakers.append(Speaker(mean,prefix))
    return speakers


def main ():
    le = preprocessing.LabelEncoder()
    mfcc_all_dictionary = dict()
    get_all_mfcc(mfcc_all_dictionary)
    speakers = get_all_speakers()
    # print(mfcc_all_dictionary)
    mfcc_means_dictionary = dict()
    # global_mfcc_means = get_all_mfcc_means(mfcc_means_dictionary)[0]
    # print(mfcc_means_dictionary)
    print("to do knn \n", len(get_all_mfcc_means(mfcc_means_dictionary)[1]))
    # knn = Knn(get_all_mfcc_means(mfcc_means_dictionary)[1])
    knn = Knn(speakers)
    # lista osób, które wygrały dla poszczególnych indeksów mfcc
    winners = knn.find_winners(speakers[0].mfcc_mean)
    # print(winners)
    # print(knn.get_winner(winners))

    knn.draw_mistake_matrix(speakers, prefixes)
    # Transposing list to match plt format
    # plt.plot(np.array(global_mfcc_means).T.tolist())
    # plt.ylabel('MFCC val')
    # plt.xlabel('MFCC idx')
    # plt.show()

# [BJ] mfcc_means: [-388.3489471435547, 49.38297576904297, 7.8444747686386105, 41.1297794342041, -0.16219628229737282, -0.0397356778383255, -1.5765647903084754, 4.141047525405884, 2.1754455413669347, 5.320105957984924, 2.952728569507599, 5.791269254684448, 1.2452065262012184]
# [HN] mfcc_means: [-195.9180160522461, 16.14417428970337, 4.962639139592648, 58.52317314147949, -25.354970359802245, 23.945265197753905, -6.618231773376465, -8.897641658782959, 6.040571284294129, 7.052741050720215, 2.785193556547165, 2.3840505599975588, 4.443789267539978]
# [PK] mfcc_means: [-192.74744567871093, 23.47176742553711, 8.000540471076965, 46.61904335021973, -5.315030300617218, 13.563743019104004, -4.909115529060363, 6.4787770986557005, -1.7978293180465699, 4.877735304832458, 1.9641524970531463, 5.268310832977295, -0.33808462498709557]
# [PT] mfcc_means: [-187.44305725097655, 28.717227935791016, 4.88146835565567, 47.61230621337891, -17.56527328491211, 13.664232444763183, -1.159687888622284, 0.7071892291307449, -1.8709264546632767, 10.436312103271485, 3.4967542767524717, -0.44094045981764796, 1.3219014942646026]
# [ZZ] mfcc_means: [-268.9265930175781, 13.547713041305542, -5.369972674548626, 40.1517147064209, -6.269543981552124, 13.892732334136962, -5.140625596046448, 6.523916959762573, 1.3403185039758683, 3.3588162899017333, 8.3439462184906, 1.376888209581375, 8.322170829772949]

if __name__ == '__main__':
    main()
