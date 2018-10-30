"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum




class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
    def is_non_terminal(self,s):
        return s.isupper()
    def is_terminal(self,s):
        return s.islower() or not s.isalpha()

    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        epsilon = 0.0001
        for key,rule_list in self.lhs_to_rules.items():
            prob_list = list(map(lambda t: t[2],rule_list))
            if abs(fsum(prob_list)-1)>epsilon:
                #print("Prob:", fsum(prob_list))
                return False

            if not self.is_non_terminal(key):
                #print ("Here",key)
                return False
            for rule in rule_list:
                t = rule[1]
                if len(t)>2:
                    print (t)
                    return False
                elif len(t)==2:
                    if not self.is_non_terminal(t[0]) or not self.is_non_terminal(t[1]):
                        print(t)
                        return False
                else: # len(t) == 1
                    if not self.is_terminal(t[0]):
                        #print("Here", t)
                        return False

        return True


if __name__ == "__main__":
    with open(sys.argv[1],'r') as grammar_file:
        grammar = Pcfg(grammar_file)

    if grammar.verify_grammar():
        print ("Great! The grammar is good.")
    else:
        print ("Sorry, grammar illegal.")
        exit()

        
