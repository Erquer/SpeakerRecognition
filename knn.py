import collections

from Name_combined_value import Name_combined_value
import numpy as np

class Knn:
    speakers = []
    k = 3
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
                local_winners = self.find_k_nearest(values_for_certain_index, value[x])
                print(local_winners)
                winners.append(local_winners)
        return winners

    def find_k_nearest(self, array, value):
        # array2 = np.asarray(list(map(lambda x: x.mfcc, array)))
        # idx = (np.abs(array2 - value))
        # return array[idx]

        for element in array:
            element.set_diff(abs(element.mfcc - value))
        sorted_winners_by_score = sorted(array, key=lambda x: x.diff, reverse= False)

        if self.k > len(sorted_winners_by_score):
            return list(map(lambda x: x.name, sorted_winners_by_score))
        elif self.k < 0:
            return []
        else:
            return list(map(lambda x: x.name, sorted_winners_by_score))[:self.k]

    def get_winner(self, winners):
        counter = 0
        if isinstance(winners[0], collections.Sequence):
            new_winners = []
            for names in winners:
                local_counter = 0
                name = names[0]
                for i in names:
                    curr_frequency = winners.count(i)
                    if curr_frequency > local_counter:
                        local_counter = curr_frequency
                        name = i
                new_winners.append(name)
            winners = new_winners
        num = winners[0]

        for i in winners:
            curr_frequency = winners.count(i)
            if (curr_frequency > counter):
                counter = curr_frequency
                num = i

        return num