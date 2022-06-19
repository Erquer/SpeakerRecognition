
class Speaker:
    mfcc_mean = []
    name = ""

    def __init__(self, mfcc_mean, name):
        self.mfcc_mean = mfcc_mean
        self.name = name