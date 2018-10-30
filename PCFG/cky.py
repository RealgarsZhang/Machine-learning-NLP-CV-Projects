"""
COMS W4705 - Natural Language Processing - Fall 2018
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def prod_map(self,list1,list2):
        res = []
        for t1 in list1:
            for t2 in list2:
                rule_list = self.grammar.rhs_to_rules[(t1,t2)]
                res += list(map(lambda t: t[0],rule_list))
        return res
    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        n = len(tokens)
        table = []
        for i in range(n+1):
            temp = []
            for j in range(n+1):
                temp.append([])
            table.append(temp)

        for i in range(n):
            token = tokens[i]
            rule_list = self.grammar.rhs_to_rules[(token,)]
            for item in rule_list:
                table[i][i+1].append(item[0])

        for length in range(2,n+1):
            for i in range(n-length+1):
                j = i+length
                for k in range(i+1,j):
                    table[i][j] += self.prod_map(table[i][k],table[k][j])

        return self.grammar.startsymbol in table[0][n]



       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = defaultdict(lambda: defaultdict(tuple))
        probs = defaultdict(lambda: defaultdict(lambda: float("-inf")))
        n = len(tokens)
        if not self.is_in_language(tokens):
            return table, probs

        for i in range(n):
            token = tokens[i]
            rule_list = self.grammar.rhs_to_rules[(token,)]
            for item in rule_list:
                table[(i,i+1)][item[0]] = token
                probs[(i,i+1)][item[0]] = math.log(item[2])

        for length in range(2,n+1):
            for i in range(n-length+1):
                j = i + length
                for k in range(i+1,j):
                    for B in probs[(i,k)]:
                        for C in probs[(k,j)]:
                            rule_list = self.grammar.rhs_to_rules[(B,C)]
                            for rule in rule_list:
                                A = rule[0]
                                temp_prob = math.log(rule[2])+probs[(i,k)][B]+probs[(k,j)][C]
                                if probs[(i,j)][A]<temp_prob:
                                    probs[(i,j)][A] = temp_prob
                                    table[(i,j)][A] = ((B,i,k),(C,k,j))

        return table, probs


def get_tree(table, i,j,nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if (i,j) not in table or nt not in table[(i,j)]:
        return None

    back_ptr = table[(i,j)][nt]
    #print (back_ptr)
    if type(back_ptr) == str:
        return(nt,back_ptr)
    res1 = get_tree(table,back_ptr[0][1],back_ptr[0][2],back_ptr[0][0])
    res2 = get_tree(table,back_ptr[1][1],back_ptr[1][2],back_ptr[1][0])
    return (nt,res1,res2)

 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks =['flights', 'from', 'miami', 'to', 'cleveland', '.']
        print(parser.is_in_language(toks))
        table,probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        assert check_probs_format(probs)
        #print (table)
        #print (probs)
        print (get_tree(table,0,len(toks),grammar.startsymbol))
