import pickle
from info import classify, MyDict
import info

with open("reduceddata.pickle", "rb") as f:
    info.pos, info.neg, info.totals = pickle.load(f)


info.features = set(info.pos.keys())|set(info.neg.keys())  


print( info.classify('great') )  