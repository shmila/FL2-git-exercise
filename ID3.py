import numpy as np
import copy
import csv


def entropy(labels,num_total):
    ent=0
    for label in labels:
        p = float(labels[label])/num_total
        ent += -p*np.log2(p)
    return ent


def get_labels(examples,target_attr):
    labels={}
    for ex in examples:
        val = ex[target_attr]
        if val in labels:
            labels[val] += 1
        else:
            labels[val] = 1
    return labels


def most_common_label(labels):
    mcl = max(labels,key=lambda k: labels[k])
    return mcl

def get_subset_examples(examples,attr):
    attr_vals = get_labels(examples,attr)
    relevant_examples = {}
    for val in attr_vals:
        relevant_examples[val] = []
    for ex in examples:
        val = ex[attr]
        relevant_examples[val].append(ex)
    return relevant_examples

def info_gain(examples,target_attr,attr):

    n = len(examples)
    attr_vals_counters = get_labels(examples,attr)
    # general entropy to start substract from
    target_labels = get_labels(examples,target_attr)
    ent_all = entropy(target_labels,n)
    # info gain value
    ig = ent_all
    # divide examples per attribute value
    relevant_examples = get_subset_examples(examples,attr)

    # calculate the entropy
    for val in attr_vals_counters:
        p = float(attr_vals_counters[val])/n
        sub_examples = relevant_examples[val]
        lbls = get_labels(sub_examples,target_attr)
        ent = entropy(lbls,len(sub_examples))
        ig -= p*ent
    return ig



def id3(examples,target_attr,remain_attrs):
        labels = get_labels(examples,target_attr)

        root = Node()

        # if only 1 label for all examples
        if len(labels.keys()) == 1:
            root.label = labels.keys()[0]
            return root
        # no more attrs left
        if len(remain_attrs)==0:
            root.label = most_common_label(labels)
            return root
        best_attr = None
        best_ig = 0
        for attr in remain_attrs:
            ig = info_gain(examples,target_attr,attr)
            if ig > best_ig:
                best_attr = attr
                best_ig = ig
        root.label = best_attr
        attr_vals = get_labels(examples,best_attr)
        sub_examples = get_subset_examples(examples,best_attr)
        for val in attr_vals:
            val_examples = sub_examples[val]
            child = Node(val)
            root.children.append(child)
            if len(val_examples) == 0:
                lbls = get_labels(val_examples,target_attr)
                grandson = Node(most_common_label(lbls))
                child.children.append(grandson)
            else:
                new_remain_attr = copy.deepcopy(remain_attrs)
                new_remain_attr.remove(best_attr)
                grandson = id3(val_examples,target_attr,new_remain_attr)
                child.children.append(grandson)

        return root

def get_rank_list(root,ind=0,rank_list=[]):

    if len(rank_list)-1<ind:
        # this is the first label in this rank
        rank_list.append([])

    rank_list[ind].append(root.label)

    if len(root.children)==0:
        return rank_list

    for child in root.children:
        rank_list = get_rank_list(child,ind+1,rank_list)
    return rank_list


class Node(object):

    def __init__(self,label=None):
        self.label=label
        self.children = []


if __name__ == "__main__":

    path = 'credithistory.csv'
    fs = csv.reader(open(path))
    all_row = []
    for r in fs:
        all_row.append(r)

    headers = all_row[0]
    examples = []
    for i in range(1,len(all_row)):
        ex = {}
        row = all_row[i]
        for j in range(len(row)):
            val = row[j]
            key = headers[j]
            ex[key] = val
        examples.append(ex)

    target_attr = 'risk'
    headers.remove(target_attr)
    root = id3(examples,target_attr,headers)
    r_list = get_rank_list(root)
    print(r_list)

    #took the examples from https://github.com/tofti/python-id3-trees