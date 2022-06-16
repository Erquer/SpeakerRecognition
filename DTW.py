import librosa
import numpy as np
import matplotlib.pyplot as plt

hop_len = 512
sam_rate = 16000
prefixes = ['BJ', 'HN', 'PK', 'PT', 'ZZ']


def main():
    record_dictionary = dict()
    mfcc_dictionary = dict()
    for prefix in prefixes:
        for i in range(1, 11):
            filename = f'./dataset/pocięte/{prefix}_{i}.wav'
            y, sr = librosa.load(filename, sr=sam_rate)
            record_dictionary[f'{prefix}_{i}'] = y
            mfcc_dictionary[f'{prefix}_{i}'] = librosa.feature.mfcc(y, sr=sam_rate, n_mfcc=13)
    noise = np.random.rand(mfcc_dictionary['BJ_1'].shape[0], 93)
    Y = np.concatenate((noise, noise, mfcc_dictionary['BJ_1'], noise), axis=1)
    D, wp = librosa.sequence.dtw(mfcc_dictionary['BJ_1'], Y, subseq=True)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    # img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
    #                                ax=ax[0])
    ax[0].set(title='DTW cost', xlabel='Noisy sequence', ylabel='Target')
    ax[0].plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    ax[0].legend()
    # fig.colorbar(img, ax=ax[0])
    ax[1].plot(D[-1, :] / wp.shape[0])
    ax[1].set(xlim=[0, Y.shape[1]], ylim=[0, 2],
              title='Matching cost function')

    plt.show()



if __name__ == '__main__':
    main()