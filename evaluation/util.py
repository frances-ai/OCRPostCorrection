# Compute character error rate (CER)
import Levenshtein

def get_cer(prediction, target):
    distance = Levenshtein.distance(prediction, target)
    return distance / len(target)





