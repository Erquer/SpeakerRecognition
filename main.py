import librosa
import matplotlib.pyplot as plt
import numpy as np

SAMPLING_RATE = 16000
hop_len = 512

def get_mfcc(prefix):
    mfccs = []
    for i in range(1,11):
        filename = f'./dataset/pocięte/{prefix}_{i}.wav'
        y, sr = librosa.load(filename, sr=SAMPLING_RATE)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mean_mfcc = mfcc.mean(axis=1)
        mfccs.append(mean_mfcc)
        if prefix == 'BJ':
            print(mean_mfcc)
    return mfccs

def main ():
    prefixes = ['BJ', 'HN']
    # filename = './dataset/pocięte/BJ_1.wav'
    # y, sr = librosa.load(filename, sr=SAMPLING_RATE)
    #
    # sampling_rate = librosa.get_samplerate(filename)
    #
    # print(f'y?? = {y}, sampling rate = {sr}, sampling_rate = {sampling_rate}')
    #
    # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    # print('Estimated tempo: {:.2f} beats per minute'.format(tempo))
    #
    # # 4. Convert the frame indices of beat events into timestamps
    # beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    # print(f'beat times = {beat_times}')
    #
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # mean_mfcc = mfcc.mean(axis=1)
    # print(f'{len(mfcc[0])}, {mean_mfcc}')
    mfcc = dict()
    for prefix in prefixes:
        mfcc[prefix] = get_mfcc(prefix)

    mfcc_tables = mfcc['BJ']
    print(len(mfcc_tables[0]))
    plt.plot([table for table in mfcc_tables])
    plt.ylabel('MFCC val')
    plt.xlabel('MFCC idx')
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
