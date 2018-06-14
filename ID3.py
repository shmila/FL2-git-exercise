import numpy as np

POSITIVE_VAL = 1
NEGATIVE_VAL = -1

def conditional_entropy(positive,negative):
    total = positive+negative
    p_portion = float(positive)/total
    n_portion = float(negative)/total
    entropy = -p_portion*np.log(p_portion) - n_portion*np.log(n_portion)
    return  entropy

def id3_algo(examples,target_attr,attrs,label_tag,parent=None):
    """
    assume example is a dictionary with the attribute:value pairs
    :param examples:
    :param target_attr:
    :param attrs:
    :return:
    """

    if not examples:
        #examples are empty
        if parent:
            return Node('leaf',parent.label,parent)
        else:
            return Node('leaf',None,None)

    root = Node('root',0,parent)
    positive_counter = 0
    negative_counter = 0
    for ex in examples:
        label = ex[label_tag]
        if label == POSITIVE_VAL:
            positive_counter += 1
        elif label == NEGATIVE_VAL:
            negative_counter += 1
    total = positive_counter+negative_counter




class Node(object):

    def __init__(self,text,label,parent=None):
        self.text = text
        self.label=label
        self.parent = parent



