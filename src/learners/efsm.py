# Copyright (c) Michal Soucha

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import Counter

LABEL_FORALL = "forall" # for any input/output
LABEL_EXISTS = "!!1" # there is exactly one
LABEL_SUBSETS = "subsets" # several groups of inputs
LABELS = [LABEL_FORALL, LABEL_SUBSETS, LABEL_EXISTS]
VARIABLE = "??" # prefix of a variable

REWARD_NEGATIVE = -1
REWARD_NEUTRAL = 0
REWARD_POSITIVE = 1
REWARD_NONNEGATIVE = 2

class Transition:
    def __init__(self,x,y,reward=None):
        self.input = x
        self.output = y
        self.reward = reward

    def __str__(self):
        return "{0}/{1} ({2})".format(self.input,self.output,self.reward)

    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash((self.input, self.output, self.reward))

    def isSpecializationOf(self, other):
        if self.input == other.input and self.output == other.output:
            return False
        if not other.reward or not self.reward: # root
            return not other.reward
        if other.input == LABEL_SUBSETS or self.input == LABEL_SUBSETS: # output == LABEL_EXISTS
            return other.input == LABEL_SUBSETS
        if other.input == LABEL_FORALL:
            if self.input == LABEL_EXISTS or (self.output != LABEL_EXISTS and len(self.output) > 1):
                return False
            return other.output == LABEL_EXISTS or other.output == self.output
        if self.input == LABEL_FORALL:
            return False
        if other.input == LABEL_EXISTS: # output == LABEL_EXISTS
            if self.input == LABEL_EXISTS or len(self.input) != len(self.output):
                return False
            diffOuts = set(self.output)
            return len(diffOuts) == len(self.output)
        if self.input == LABEL_EXISTS:
            return False
        if len(self.input) > len(other.input) or len(self.output) > len(other.output):
            return False
        if len(self.output) == len(other.output) == 1:
            if self.output != other.output:
                return False
            for i in self.input:
                if i not in other.input:
                    return False
            return True
        # compare mappings
        for i in range(len(self.input)-1):
            idx = other.input.find(self.input[i])
            if idx == -1 or self.output[idx] != other.output[idx]:
                return False
        return True

class EFSM:
    def __init__(self, isFSM=False):
        self.transitions = {}
        self.negativeRewardTransitions = {}
        self.isFSM = isFSM

    def __str__(self):
        desc = "{0}-state EFSM\n".format(len(self.transitions))
        for state in self.transitions:
            for tran in self.transitions[state]:
                desc += "{0} -{1}-> {2}\n".format(state, tran, self.transitions[state][tran])
            if state in self.negativeRewardTransitions:
                desc += "{0} neg:".format(state)
                for tran in self.negativeRewardTransitions[state]:
                    desc += " {0}".format(tran)
            desc += "\n"
        return desc

    def __repr__(self):
        return self.__str__()

    def writeDOT(self, dotFile, id):
        if self.transitions:
            for state in self.transitions:
                for t, ns in self.transitions[state].items():
                    print('  g{0}s{1} -> g{0}s{2} [label="{3}/{4} {5}"];'.format(id, state, ns, t.input, t.output, t.reward), file=dotFile)
        else:
            print('  g{0}s0;'.format(id), file=dotFile)
                
    def addTransition(self, state, transition, nextState):
        if transition.reward != REWARD_NEGATIVE:
            if state in self.transitions:
                if transition in self.transitions[state]:
                    return self.transitions[state][transition]
                else:
                    self.transitions[state][transition] = nextState
            else:
                self.transitions[state] = {transition: nextState}
        else:
            if state in self.negativeRewardTransitions:
                if transition in self.negativeRewardTransitions[state]:
                    return self.negativeRewardTransitions[state][transition]
                else:
                    self.negativeRewardTransitions[state][transition] = nextState
            else:
                self.negativeRewardTransitions[state] = {transition: nextState}

    def removeTransition(self, state, transition):
        if transition.reward != REWARD_NEGATIVE:
            if state in self.transitions:
                if transition in self.transitions[state]:
                    self.transitions[state].pop(transition)
                    if not self.transitions[state]:
                        self.transitions.pop(state)
        else:
            if state in self.negativeRewardTransitions:
                if transition in self.negativeRewardTransitions[state]:
                    self.negativeRewardTransitions[state].pop(transition)
                    if not self.negativeRewardTransitions[state]:
                        self.negativeRewardTransitions.pop(state)

    def getTransition(self, state, x, y=None, reward=None):
        if (not reward or reward != REWARD_NEGATIVE) and state in self.transitions:
            for tran in self.transitions.get(state):
                if (not x or x == tran.input or tran.input in LABELS or x in tran.input) and \
                    (not y or y == tran.output or tran.output == LABEL_EXISTS or y == LABEL_EXISTS or y in tran.output) and \
                    (not reward or reward == REWARD_NONNEGATIVE or reward == tran.reward):
                    return tran
        if (not reward or reward == REWARD_NEGATIVE) and state in self.negativeRewardTransitions:
            for tran in self.negativeRewardTransitions.get(state):
                if (not x or x == tran.input or tran.input in LABELS or x in tran.input) and \
                    (not y or y == tran.output or tran.output == LABEL_EXISTS or y == LABEL_EXISTS or y in tran.output):
                    return tran
        return None
        
    def getOutputs(self, state, reward, x=None):
        outputs = set()
        if not reward or reward != REWARD_NEGATIVE:
            if state in self.transitions:
                for tran in self.transitions.get(state):
                    if (not x or (tran.input not in LABELS and x in tran.input)) and tran.reward \
                        and (not reward or reward == REWARD_NONNEGATIVE or reward == tran.reward):
                        for o in tran.output:
                            outputs.add(o)
        if not reward or reward == REWARD_NEGATIVE:
            if state in self.negativeRewardTransitions:
                for tran in self.negativeRewardTransitions.get(state):
                    if not x or x == tran.input:
                        for o in tran.output:
                            outputs.add(o)
        return outputs

    def getNonnegativeRewardOutputWithReward(self, state, x=None):
        if state in self.transitions:
            for tran in self.transitions.get(state):
                if (not x and tran.input != LABEL_FORALL) \
                    or (x and (tran.input in LABELS or x in tran.input)):
                    if not x or len(tran.input) == 1 or tran.input in LABELS:
                        return tran.output, tran.reward
                    return tran.output[tran.input.find(x)], tran.reward # mapping
        return None, None

    def getNonnegativeRewardOutput(self, state, x=None):
        if state in self.transitions:
            for tran in self.transitions.get(state):
                if (not x and tran.input != LABEL_FORALL) \
                    or (x and (tran.input in LABELS or x in tran.input)):
                    if not x or len(tran.input) == 1 or tran.input in LABELS:
                        return tran.output
                    return tran.output[tran.input.find(x)] # mapping
        return None

    def getNextState(self, state, tran):
        if tran.reward != -1:
            if state in self.transitions:
                if tran in self.transitions[state]:
                    return self.transitions.get(state).get(tran)
                for t in self.transitions[state]:
                    if t.input in LABELS or tran.input in t.input:
                        if tran.reward == t.reward and (t.output == LABEL_EXISTS or tran.output in t.output):
                            return self.transitions[state][t]
                        break
        else:
            if state in self.negativeRewardTransitions:
                if tran in self.negativeRewardTransitions[state]:
                    return self.negativeRewardTransitions.get(state).get(tran)
                for t in self.negativeRewardTransitions[state]:
                    if t.input in LABELS or tran.input in t.input:
                        if t.output == LABEL_EXISTS or tran.output in t.output:
                            return self.transitions[state][t]
        return None

    def getEndPathState(self, state, path):
        for input in path:
            tran = self.getTransition(state, input)
            if not tran:
                return None
            nextState = self.getNextState(state, tran)
            if not nextState:
                return None
            state = nextState
        return state

    def getAccessSequence(self, state):
        if state == 0:
            return ""
        #TODO
        return ""

    def hasNonnegativeRewardTransition(self):
        return len(self.transitions) > 0

    def hasConflictingTransition(self, other):
        # TODO bfs comparison
        for state in self.transitions:
            if state in other.transitions:
                for tran in self.transitions[state]:
                    for oTran in other.transitions[state]:
                        if tran.input == oTran.input and \
                            ((tran.output == oTran.output and tran.reward != oTran.reward) or \
                            (tran.output != oTran.output and tran.reward != -1 and oTran.reward != -1)):
                                return True
        return False

    def join(self, other):
        for state in other.transitions:
            if state not in self.transitions:
                self.transitions[state] = {}
            self.transitions[state].update(other.transitions[state])

    def hasTransitionWithDifferentReward(self, state, output, reward):
        for tran in self.transitions[state]:
            if output in tran.output and tran.reward != reward:
                return True
        return False

    def tryGeneralize(self, parentEFSM):
        for state in self.transitions:
            if len(self.transitions[state]) > 1:
                mapping = {}
                for tran, nextState in self.transitions[state].items():
                    mapping[(tran.reward, tran.output, nextState)] = \
                        mapping.get((tran.reward, tran.output, nextState), "") + tran.input
                if len(mapping) == 1:
                    ((reward, output, nextState), inputs) = list(mapping.items())[0]
                    if len(inputs) > 2:
                        genEFSM = EFSM()
                        tran = Transition(LABEL_FORALL, output, reward)                        
                        #TODO add not just one transition
                        genEFSM.addTransition(state, tran, nextState)
                        if genEFSM.isSpecializationOf(parentEFSM):
                            return [genEFSM]
                else:
                    genEFSM = EFSM()
                    singleInputs = True
                    for (reward, output, nextState), inputs in mapping.items():
                        if len(inputs) > 1:
                            singleInputs = False
                        tran = Transition(inputs, output, reward)                        
                        #TODO add not just one transition
                        genEFSM.addTransition(state, tran, nextState)
                    targets = set([(el[0],el[2]) for el in mapping.keys()])
                    if singleInputs:
                        genEFSM = None
                        if len(mapping) > 2:
                            if len(targets) == 1:
                                (reward, nextState) = targets.pop()
                                genEFSM = EFSM()
                                tran = Transition(LABEL_EXISTS, LABEL_EXISTS, reward)
                                genEFSM.addTransition(state, tran, nextState)
                                targets.add((reward, nextState))
                            elif len(targets) < len(mapping):
                                for reward, nextState in targets:
                                    ioMap = [(input, output) for (r, output, ns), input in mapping.items() if r == reward and ns == nextState]
                                    inputs = "".join([el[0] for el in ioMap])
                                    outputs = "".join([el[1] for el in ioMap])
                                    tran = Transition(inputs, outputs, reward)
                                    genEFSM.addTransition(state, tran, nextState)
                    if not genEFSM or genEFSM.isSpecializationOf(parentEFSM):
                        moreGenEFSM = EFSM()
                        for reward, nextState in targets:
                            tran = Transition(LABEL_SUBSETS, LABEL_EXISTS, reward)
                            moreGenEFSM.addTransition(state, tran, nextState)
                        if moreGenEFSM.isSpecializationOf(parentEFSM):
                            if genEFSM:
                                return [genEFSM, moreGenEFSM]
                            else:
                                return [moreGenEFSM]
                        elif genEFSM:
                            return [genEFSM]
        return []
                        
    def tryGeneralizeWith(self, other, parentEFSM):
        if self.isFSM or other.isFSM:
            return
        for state in self.transitions:
            if state in other.transitions:
                trans = [(t, ns) for (t, ns) in self.transitions[state].items() if t.reward != -1]
                oTrans = [(t, ns) for (t, ns) in other.transitions[state].items() if t.reward != -1]
                if trans[0][0].input in LABELS or oTrans[0][0].input in LABELS:
                    tran, nextState = trans[0]
                    oTran, oNextState = oTrans[0]
                    if tran.input == LABEL_FORALL or oTran.input == LABEL_FORALL:
                        if tran.input == LABEL_FORALL and oTran.input == LABEL_FORALL:
                            if nextState == oNextState and tran.reward == oTran.reward:
                                genEFSM = EFSM()
                                #TODO add not just one transition
                                genEFSM.addTransition(state, Transition(LABEL_FORALL, LABEL_EXISTS, tran.reward), nextState)
                                if genEFSM.isSpecializationOf(parentEFSM):
                                    return [genEFSM]
                        else:
                            uniqueOut = (tran.reward == oTran.reward and nextState == oNextState)
                            if uniqueOut:
                                if tran.input == LABEL_FORALL:
                                    tmpTrans = oTrans
                                else:
                                    tmpTrans = trans
                                if uniqueOut and tmpTrans[0][0].input != LABEL_EXISTS:
                                    for t, ns in tmpTrans:
                                        if len(t.output) != 1 or t.reward == tran.reward or ns != nextState:
                                            uniqueOut = False
                                            break
                                if uniqueOut:
                                    genEFSM = EFSM()
                                    #TODO add not just one transition
                                    genEFSM.addTransition(state, Transition(LABEL_SUBSETS, LABEL_EXISTS, tran.reward), nextState)
                                    if genEFSM.isSpecializationOf(parentEFSM):
                                        return [genEFSM]
                    #elif: #other labels
               # else: # mappings
                #if len(transRew0) + len(transRew1) == len(oTransRew0) + len(oTransRew1) == 1:
                #    if len(transRew1) != len(oTransRew1):
                #        if len(transRew1) == 1:
                #            tran, nextState = transRew1[0]
                #            oTran, oNextState = oTransRew0[0]
                #        else:
                #            tran, nextState = transRew0[0]
                #            oTran, oNextState = oTransRew1[0]
                #        if not self.hasTransitionWithDifferentReward(state, oTran.output, oTran.reward) and \
                #            not other.hasTransitionWithDifferentReward(state, tran.output, tran.reward):
                #            genEFSM = EFSM()
                #            #TODO add not just one transition
                #            genEFSM.addTransition(state, tran, nextState)
                #            genEFSM.addTransition(state, oTran, oNextState)
                #            if genEFSM.isSpecializationOf(parentEFSM):
                #                return [genEFSM]
                #        continue
                #    if len(transRew1) == 0:
                #        oTransRew1 = oTransRew0
                #        transRew1 = transRew0
                #    tran, nextState = transRew1[0]
                #    oTran, oNextState = oTransRew1[0]
                    
                #else:
                    
                
          
    def isSpecializationOf(self, other, X=set()):
        # simulation
        # TODO check next states
        hasSpecializedTransition = False
        for state in self.transitions:
            if state not in other.transitions:
                return False
            uniqueOutput = None
            uniqueOutputForEach = set()
            for tran in self.transitions[state]:
                t = other.getTransition(state, tran.input, tran.output)
                if t:
                    if t.reward and tran.reward != t.reward:
                        return False
                    if uniqueOutput and uniqueOutput != tran.output:
                        return False
                    if tran.input != t.input or tran.output != t.output:
                        if not tran.isSpecializationOf(t):
                            return False
                        hasSpecializedTransition = True
                    if not uniqueOutput and t.input == LABEL_FORALL and (t.output == LABEL_EXISTS or len(t.output) == 1) \
                        and tran.output != LABEL_EXISTS:
                        uniqueOutput = tran.output 
                    elif t.input == t.output == LABEL_EXISTS:
                        for o in tran.output:
                            if o in uniqueOutputForEach:
                                return False
                            uniqueOutputForEach.add(o)
                else:
                    return False # not covered
            if uniqueOutput and self.getTransition(state, None, uniqueOutput, -1):
                # there is transition x/uniqueOutput with reward -1 that contradicts other's LABEL_FORALL/LABEL_EXISTS
                return False
        for state in self.negativeRewardTransitions:
            if state not in other.transitions:
                return False
            outputs = set()
            inputs = Counter()
            for tran in self.negativeRewardTransitions[state]:
                t = other.getTransition(state, tran.input)
                if t:
                    if t.reward and t.reward != -1 and t.output != LABEL_EXISTS and tran.output in t.output:
                        return False
                    if t.input == LABEL_FORALL:
                        if t.output == LABEL_EXISTS:
                            outputs.add(tran.output)
                    elif t.input == LABEL_EXISTS == t.output:
                        inputs[tran.output] += 1
            if X:
                if (outputs and X == outputs) or (inputs and inputs.most_common(1)[0][1] == len(X)):
                    return False
        return len(self.transitions) == 0 or hasSpecializedTransition      
