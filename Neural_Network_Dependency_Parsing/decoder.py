from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def is_doable(self,op_pair,state):
        if op_pair[0] == "shift":
            if state.stack and len(state.buffer)==1:
                return False
            else:
                item = state.buffer.pop()
                state.stack.append(item)
                return True
        elif op_pair[0] == "left_arc":
            if len(state.stack) == 0  or state.stack[-1] == 0:
                return False
            else:
                parent = state.buffer[-1]
                child = state.stack.pop()
                relation = op_pair[1]
                state.deps.add((parent,child,relation))
                return True
        else: #"right_arc"
            if len(state.stack)==0:
                return False
            else:
                parent = state.stack.pop()
                child = state.buffer.pop()
                state.buffer.append(parent)
                relation = op_pair[1]
                state.deps.add((parent, child, relation))
                return True


    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        label_map = dict((self.extractor.output_labels[name],name) for name in self.extractor.output_labels)
        while state.buffer:
            vec = self.extractor.get_input_representation(words,pos,state)
            pred = self.model.predict(vec.reshape((1,len(vec))))
            sorted_ops = np.argsort(pred).reshape((pred.shape[1],))
            for i in range(1,len(sorted_ops)+1):
                op_pair = label_map[int(sorted_ops[-i])]
                if self.is_doable(op_pair,state):
                    break

            # TODO: Write the body of this loop for part 4 

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
