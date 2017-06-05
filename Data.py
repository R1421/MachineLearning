import gzip, cPickle

class Numbers:
    def __init__(self,location):
        f = gzip.open(location)
        train,valid,test = cPickle.load(f)
        self.train = train
        self.valid = valid
        self.test = test