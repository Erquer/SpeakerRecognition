from Name_combined_value import Name_combined_value
import numpy as np

class Knn:
    speakers = []

    def __init__(self,  speakers):
        self.speakers = speakers


    #value is vector of mfcc means for who we are looking for closest neighbour
    def find_winners(self, value):
        winners = []
        if self.speakers:

            for x in range(0, len(self.speakers[0].mfcc_mean)):
                values_for_certain_index = []
                for speaker in self.speakers:
                    values_for_certain_index.append(Name_combined_value(speaker.mfcc_mean[x], speaker.name))
                values_for_certain_index = sorted(values_for_certain_index, key=lambda x: x.mfcc, reverse=False)
                local_winner = self.find_nearest(values_for_certain_index, value[x])
                winners.append(local_winner.name)
        return winners

    def find_nearest(self, array, value):
        array2 = np.asarray(list(map(lambda x: x.mfcc, array)))
        idx = (np.abs(array2 - value)).argmin()
        return array[idx]

    def get_winner(self, winners):
        counter = 0
        num = winners[0]

        for i in winners:
            curr_frequency = winners.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                num = i

        return num