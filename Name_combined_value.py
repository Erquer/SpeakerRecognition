class Name_combined_value:
    mfcc = None
    name = ""
    index = None
    diff = None

    def __init__(self, mfcc, name):
        self.mfcc = mfcc
        self.name = name

    def set_index(self, index):
        self.index = index
    def set_diff(self, diff):
        self.diff = diff