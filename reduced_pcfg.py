import nltk
import pandas as pd
from nltk.corpus import treebank
from pprint import pprint
import numpy as np
from time import sleep
import re

def main():
	cfg = pd.read_pickle('cfg_pickle.pickle')




	cfg = cfg.to_frame().reset_index() #turns Series into dataframe

	pprint(set([val for children in cfg.children for val in children]))
	pprint(len(set([val for children in cfg.children for val in children])))




	cfg.rename(columns={0:'instances'}, inplace=True)

	reduced_cfg_1 = part_1(cfg)
	reduced_cfg_1.to_csv('reduced_cfg_part_1.csv')

	reduced_cfg_2 = part_2(cfg)
	reduced_cfg_2.to_csv('reduced_cfg_part_2')

def part_1(cfg):
	'''gives us a reduced grammar as definied by part 1'''
	


	cfg['parent'] = cfg.parent.apply(lambda x: re.split(r'[-=|]', x)[0]) #gives us our reduced parent tags






	cfg.children = cfg.children.apply(tuple) 

	cfg.children = cfg.children.apply(lambda x: tuple(re.split(r'[-=|]', token)[0] if token != '-NONE-' else token for token in x)) #reducing our child tags

	
	

	


	parent_counts = cfg.groupby('parent')['instances'].sum() #gives us the number of times our reduced parents appear

	rule_counts = cfg.groupby(['parent', 'children'])['instances'].sum() #gives the instances each reduced rule appears


	# print(for row in cfg.children)

	# print(len(rule_counts))
	reduced_pcfg = rule_counts.divide(parent_counts) #calculates our new probabilities based on the rules, and the parents

	return reduced_pcfg

def part_2(cfg):
	cfg['parent'] = cfg.parent.apply(lambda x: re.split(r'[-=|]', x)[0])

	mappings = {
	'ADJP': "ADJ",
	"ADVP": "ADV",
	"CONJP": "CONJP",
	"FRAG": "FRAG",
	"INTJ": "INTJ",
	"LST": "LST",
	"NAC": "NAC",
	"NP": "N",
	"NX": "N",
	"PP": "P",
	"PRN": "N",
	'PRT': 'PRT',
	'QP': 'QP',
	'RRC': 'RRC',
	'S': 'S',
	"SBAR": "S",
	'SBARQ': 'S',
	'SINV': 'S',
	'SQ' : 'S',
	'UCP': 'UCP',
	'VP': 'V',
	'WHADJP': 'ADJ',
	'WHADVP': 'V',
	'WHNP': 'N',
	'WHPP': 'P',
	'X': 'X' 
	}
	cfg.parent = cfg.parent.map(mappings)
	parent_counts = cfg.groupby('parent')['instances'].sum()

	rule_counts = cfg.groupby(['parent', 'children'])['instances'].sum() #gives the instances each reduced rule appears

	reduced_pcfg = rule_counts.divide(parent_counts) #calculates our new probabilities based on the rules, and the parents


	return reduced_pcfg
	
if __name__ == '__main__':
	main()