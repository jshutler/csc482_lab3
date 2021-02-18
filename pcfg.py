import nltk
import pandas as pd
from nltk.corpus import treebank
from pprint import pprint
import numpy as np
from time import sleep



def main():

    #getting all treebank files
    files = treebank.fileids()


    #grammar will be stored in dict
    grammar = {}
    for file in files:
        trees = treebank.parsed_sents(file)

        for tree in trees:
            grammar = traverse_tree(tree, grammar)


    cfg = get_grammar_series(grammar)

    cfg.to_csv("cfg_instances.csv")

    cfg.to_pickle('cfg_pickle.pickle')



    pcfg = get_probabilities(grammar)

    
    

    pcfg_series = get_grammar_series(pcfg)
    
    pcfg_series.to_csv('problem_1_pcfg.csv')

def traverse_tree(tree, tag_dict):
    """This function will traverse down the tree and continuoulsy append a dictionary
    with all the Grammer rules that it finds. And as it searches down the tree. The general layout
    of the dictionary is:
    {"tag": {"child_tags": count of instance}}
    """
    head_tag = tree.label()

    #if the subtree is a nonterminal, we want its head tag, if its terminal, we want the string itself
    child_tags = tuple([subtree.label() for subtree in tree if type(subtree) == nltk.tree.Tree]) #will only grab labels for non terminal subtrees
    

    num_children = len(child_tags)

    #if we've never seen this parent tag before
    if head_tag not in tag_dict.keys():
        #we only care if there are children here
        if num_children > 0:
            tag_dict[head_tag] = {child_tags: 1}

    #if we've seen this parent tag, but not the child tags associated with it
    elif child_tags not in tag_dict[head_tag].keys():
        #we only want to add rules if there are children to grab
        if num_children > 0:
            tag_dict[head_tag][child_tags] = 1

    #if we've seen this grammar rule before, update the count
    else:
        if num_children > 0:
            tag_dict[head_tag][child_tags] += 1        
   


    for subtree in tree:
        if type(subtree) == nltk.tree.Tree:
            # print('main')
            # print(tree.label(),  subtree)
            tag_dict = traverse_tree(subtree, tag_dict)

            # print('nonterminal' , subtree)
        else:
       
            return tag_dict

    return tag_dict


def get_probabilities(grammar):
    for parent_tag, inner_dict in grammar.items():
        total_parent_tag_instances = sum([val for val in inner_dict.values()])

        for child_key, instances in inner_dict.items():
            grammar[parent_tag][child_key] = instances / total_parent_tag_instances 

    return grammar


def get_grammar_series(grammar):
    '''turns grammar dict into grammar Series'''
    tuples = []
    values = []

    #getting our index and values in a nice format
    for parent, child in grammar.items():
        for child_tag, value in child.items():
            tuples.append((parent, child_tag))
            values.append(value)

    index = pd.MultiIndex.from_tuples(tuples, names=['parent', 'children'])

    return pd.Series(values, index=index)


if __name__ == '__main__':
    main()

    
