# Copyright (c) Michal Soucha

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import random
import logging
import string
import itertools
from collections import Counter
from core.serializer import StandardSerializer, IdentitySerializer
from learners.base import BaseLearner

debFile = open("gEfsm.txt", 'w')
def glog(msg, o=None):
    if o:
        print(msg)
    print(msg, file=debFile)

MIN_CONSECUTIVE_REWARDS = 5

LABEL_FORALL = "forall" # for any input/output
LABEL_EXISTS = "!!1" # there is exactly one
LABEL_SUBSETS = "subsets" # several groups of inputs
LABELS = [LABEL_FORALL, LABEL_SUBSETS, LABEL_EXISTS]
VARIABLE = "??" # prefix of a variable

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
        self.isFSM = isFSM

    def __str__(self):
        desc = "{0}-state EFSM\n".format(len(self.transitions))
        for state in self.transitions:
            for tran in self.transitions[state]:
                if tran.reward != -1:
                    desc += "{0} -{1}-> {2}\n".format(state, tran, self.transitions[state][tran])
            desc += "{0} neg:".format(state)
            for tran in self.transitions[state]:
                if tran.reward == -1:
                    desc += " {0}".format(tran)
            desc += "\n"
        return desc

    def __repr__(self):
        return self.__str__()
    
    def addTransition(self, state, transition, nextState):
        if state in self.transitions:
            if transition in self.transitions[state]:
                return self.transitions[state][transition]
                #if nextState not in self.transitions[state][transition]:
                #    self.transitions[state][transition].add(nextState)
            else:
                self.transitions[state][transition] = nextState
        else:
            self.transitions[state] = {transition: nextState}

    def removeTransition(self, state, transition):
        if state in self.transitions:
            if transition in self.transitions[state]:
                self.transitions[state].pop(transition)
                if not self.transitions[state]:
                    self.transitions.pop(state)

    def getTransition(self, state, x, y=None, reward=None):
        if state in self.transitions:
            for tran in self.transitions.get(state):
                if (not x or x == tran.input or tran.input in LABELS or x in tran.input) and \
                    (not y or y == tran.output or tran.output == LABEL_EXISTS or y == LABEL_EXISTS or y in tran.output) and \
                    (not reward or reward == tran.reward):
                    return tran
        return None
        
    def getNegativeRewardOutputs(self, state, x):
        outputs = set()
        if state in self.transitions:
            for tran in self.transitions.get(state):
                if tran.reward == -1 and x == tran.input:
                    for o in tran.output:
                        outputs.add(o)
        return outputs

    def getNonnegativeRewardOutput(self, state, x=None):
        if state in self.transitions:
            for tran in self.transitions.get(state):
                if tran.reward != -1 and ((not x and tran.input != LABEL_FORALL) \
                    or (x and (tran.input in LABELS or x in tran.input))):
                    if not x or len(tran.input) == 1 or tran.input in LABELS:
                        return tran.output
                    return tran.output[tran.input.find(x)] # mapping
        return None

    def getNextState(self, state, tran):
        if state in self.transitions:
            if tran in self.transitions[state]:
                return self.transitions.get(state).get(tran)
            for t in self.transitions[state]:
                if t.input in LABELS or tran.input in t.input:
                    if tran.reward == t.reward and (t.output == LABEL_EXISTS or tran.output in t.output):
                        return self.transitions[state][t]
                    break
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
                    if tran.reward != -1:
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
                    for (reward, output, nextState), inputs in mapping.items():
                        tran = Transition(inputs, output, reward)                        
                        #TODO add not just one transition
                        genEFSM.addTransition(state, tran, nextState)
                    if genEFSM.isSpecializationOf(parentEFSM):
                        moreGenEFSM = EFSM()
                        for reward, nextState in set([(el[0],el[2]) for el in mapping.keys()]):
                            tran = Transition(LABEL_SUBSETS, LABEL_EXISTS, reward)
                            moreGenEFSM.addTransition(state, tran, nextState)
                        if moreGenEFSM.isSpecializationOf(parentEFSM):
                            return [genEFSM, moreGenEFSM]
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
                    
                
          
    def isSpecializationOf(self, other):
        # simulation
        # TODO
        hasSpecializedTransition = False
        for state in self.transitions:
            if state not in other.transitions:
                return False
            for tran in self.transitions[state]:
                t = other.getTransition(state, tran.input, tran.output)
                if t:
                    if t.reward and tran.reward != t.reward:
                        return False
                    if tran.reward != -1 and (tran.input != t.input or tran.output != t.output):
                        if not tran.isSpecializationOf(t):
                            return False
                        hasSpecializedTransition = True                    
                elif tran.reward != -1:
                    return False # not covered
        return hasSpecializedTransition      
                
        
class EFSMhierarchyNode:
    def __init__(self, parent, fsm, id):
        self.parent = parent
        self.children = []
        self.id = id
        self.scoreUse = 1
        self.efsm = fsm
        self.currState = 0
        self.possibleStates = set()
        #self.possibleStates.add(self.currState)
        self.addedTransitions = []
        self.logger = logging.getLogger(__name__)

    def __str__(self):
        desc = "use:{3},id:{4} State {0} {5} of\n{1} with {2} children:\n".format(self.currState, self.efsm,
                                                                       len(self.children), self.scoreUse, self.id, self.possibleStates)
        for child in self.children:
            desc += "=> {0}\n".format(child)
        return desc

    def __repr__(self):
        return self.__str__()
    
## simulation functions

    def getOutput(self, FSMid, x):
        ## returns (guessed) output according to the followed grammar
        output = self.efsm.getNonnegativeRewardOutput(self.currState, x)
        if output:
            if len(output) > 1: # output is not a particular symbol
                output = self.children[-1].getOutput(FSMid, x)
        return output

    def updateCurrentPosition(self, FSMid, tran, parentTran=None):
        ## updates the current state and returns False if there is the given transition from the current state
        ##  and if the initial state is reached, returns the second True
        ## otherwise, (True, True) is returned so the learning starts
        learning = True
        initReached = True
        t = self.efsm.getTransition(self.currState, tran.input, tran.output)
        if t:
            if not t.reward or t.reward == tran.reward:
                self.currState = self.efsm.getNextState(self.currState, t)
                if tran.reward != -1:
                    self.scoreUse += 1
                if self.efsm.isFSM: # leaf
                    learning = (tran.reward == -1)
                    initReached = (self.currState == 0)
                else:
                    chLearning, chInitReached = self.children[-1].updateCurrentPosition(FSMid, tran, t)
                    learning = learning and chLearning
                    initReached = initReached and chInitReached
            else:
                #learning = True
                if tran.reward == -1:
                    self.scoreUse -= 1
        else:
            # could it be extended?
            if self.efsm.isFSM and tran.reward != -1:
                t = self.efsm.getTransition(self.currState, None, tran.output, tran.reward)
                nextState = self.efsm.getNextState(self.currState, t)
                self.efsm.addTransition(self.currState, tran, nextState)
                self.addedTransitions.append((self.currState, tran))
                self.currState = nextState
                if self.currState == 0:
                    self.addedTransitions.clear()
                learning = (tran.reward == -1)
                initReached = (self.currState == 0)
            # else: TODO
        return learning, initReached

    def removeLastAddedTransitions(self, FSMid):
        for state, tran in self.addedTransitions:
            self.efsm.removeTransition(state, tran)
        self.addedTransitions.clear()
        if not self.efsm.isFSM:
            self.children[-1].removeLastAddedTransitions(FSMid)

## learning functions

    def getEstimatedOutputs(self, x, FSMid, trace, counts):
        ## 
        estimated = False
        if len(self.possibleStates) > 0:
            if self.efsm.isFSM or not self.children[-1].getEstimatedOutputs(x, FSMid, trace, counts):
                for state in self.possibleStates:
                    output = self.efsm.getNonnegativeRewardOutput(state, x)
                    if output:
                        if len(output) == 1:
                            counts[output] += 1 # self.scoreUse
                        elif output == LABEL_EXISTS:
                            t = self.efsm.getTransition(state, x, output)
                            accessSeq = self.efsm.getAccessSequence(state)
                            if t.input == LABEL_EXISTS:
                                for child in self.children:
                                    if child.id >= FSMid:
                                        chState = child.efsm.getEndPathState(0, accessSeq)
                                        if chState:
                                            chTran = child.efsm.getTransition(chState, x, None, t.reward)
                                            if chTran and len(chTran.input) == len(chTran.output) == 1: # not a mapping
                                                if chTran.input == x:
                                                    counts[chTran.output] += 1
                                                else:
                                                    counts[chTran.output] -= 1
                                            for chOut in child.efsm.getNegativeRewardOutputs(chState, x):
                                                counts[chOut] -= 1
                                for tran in trace:
                                    if tran.input == x:
                                        if tran.reward != -1:
                                            counts[tran.output] += 1
                                        else:
                                            counts[tran.output] -= 1
                            elif t.input == LABEL_FORALL or t.input == LABEL_SUBSETS:
                                for child in self.children:
                                    if child.id >= FSMid:
                                        chState = child.efsm.getEndPathState(0, accessSeq)
                                        if chState:
                                            chTran = child.efsm.getTransition(chState, x, None, t.reward)
                                            if chTran:
                                                counts[chTran.output] += 1
                                            for chOut in child.efsm.getNegativeRewardOutputs(chState, x):
                                                counts[chOut] -= 1
                                for tran in trace:
                                    if tran.reward != -1:
                                        counts[tran.output] += 1
                                    else:
                                        counts[tran.output] -= 1
                            else:
                                output #TODO is it possible?
            estimated = (len(counts) > 0)
        return estimated

    def getMostProbableOutputs(self, x, trace, counts):
        ## returns the most probable output for the given input x
        estimated = False
        if len(self.possibleStates) > 0:
            for child in self.children:
                estimated = child.getMostProbableOutputs(x, trace, counts) or estimated
            if not estimated:
                for state in self.possibleStates:
                    output = self.efsm.getNonnegativeRewardOutput(state, x)
                    if output and len(output) == 1:
                        counts[output] += self.scoreUse
                        estimated = True;
                    else:
                        output = self.efsm.getNonnegativeRewardOutput(state)
                        if output and output != LABEL_EXISTS:
                            for o in output:
                                counts[o] += 1 # self.scoreUse
                            estimated = True;
        return estimated
        
    def getNegativeOutputs(self, x):
        negOuts = set()
        for state in self.possibleStates:
            negOuts |= self.efsm.getNegativeRewardOutputs(state, x)
        for child in self.children:
            negOuts |= child.getNegativeOutputs(x)
        return negOuts
    
    def tryGeneralizeChildren(self):
        #maxScore = 0
        node = self.children[-1]
        for child in reversed(self.children):
            if node != child:
                #if maxScore < child.scoreUse:
                #    maxScore = child.scoreUse
                genEFSMs = node.efsm.tryGeneralizeWith(child.efsm, self.efsm)
                if genEFSMs:
                    self.updateWithGeneralizedEFSMs(genEFSMs)
                    return self.tryGeneralizeChildren()
        #node.scoreUse = maxScore + 1

    def updateWithGeneralizedEFSMs(self, genEFSMs):
        generalizedEFSM = genEFSMs.pop()
        hierNode = EFSMhierarchyNode(self, generalizedEFSM, 0)
        updatedChildren = []
        for child in self.children:
            if child.efsm.isSpecializationOf(generalizedEFSM):
                hierNode.children.append(child)
                child.parent = hierNode
                if hierNode.id < child.id:
                    hierNode.id = child.id    
            else:
                updatedChildren.append(child)
        updatedChildren.append(hierNode)
        self.children[:] = updatedChildren
        if genEFSMs:
            hierNode.updateWithGeneralizedEFSMs(genEFSMs)
        else:
            hierNode.tryGeneralizeChildren()
        self.tryGeneralizeChildren()

    def updateWithNewEFSM(self, hypothesis, newId):
        if self.efsm.isFSM: # leaf
            if not self.efsm.hasConflictingTransition(hypothesis):
                self.efsm.join(hypothesis)
                self.scoreUse += 1
                return self
        elif hypothesis.isSpecializationOf(self.efsm):
            node = None
            for child in self.children:
                node = child.updateWithNewEFSM(hypothesis, newId)
                if node:
                    if not node.efsm.isFSM:
                        #node.scoreUse = max(ch.scoreUse for ch in self.children) + 1
                        if self.id < node.id:
                            self.id = node.id
                        return self
                    break
            if not node:
                node = EFSMhierarchyNode(self, hypothesis, newId)
                self.children.append(node)
            if self.id < node.id:
                self.id = node.id
            genEFSMs = node.efsm.tryGeneralize(self.efsm)
            if genEFSMs:
                self.updateWithGeneralizedEFSMs(genEFSMs)
            else:
                self.tryGeneralizeChildren()
            return self
        return None
            
    def initPossiblePossitions(self, trace):
        self.possibleStates.clear()
        self.possibleStates.add(0)
        # TODO
        for tran in trace:
            t = self.efsm.getTransition(0, tran.input, tran.output)
            if t.reward != tran.reward and tran.output == t.output:
                self.possibleStates.discard(0)
                break
        if self.possibleStates:
            for child in self.children:
                child.initPossiblePossitions(trace)

    def updatePossiblePositions(self, trace, possibleFSMs):
        ##
        tran = trace[-1]
        initialReached = False
        if self.possibleStates:
            for child in self.children:
                initialReached = child.updatePossiblePositions(trace, possibleFSMs) \
                    or initialReached
            statesToRemove = []
            statesToAdd = []
            for state in self.possibleStates:
                nextState = self.efsm.getNextState(state, tran)
                #glog("upp {0}-{1}->{2} ".format(state, tran, nextState))
                if nextState is not None: # comparison with None as nextState could be 0 -> False
                    #if state != 0:
                    statesToRemove.append(state)
                    statesToAdd.append(nextState)
                    if tran.reward != -1:
                        self.scoreUse += 1
                else:
                    t = self.efsm.getTransition(state, tran.input, tran.output)
                    if t and t.reward and t.reward != tran.reward:
                        statesToRemove.append(state)
                        self.scoreUse -= 1
                    #glog(" {0}".format(t))
            for state in statesToRemove:
                self.possibleStates.discard(state)
            for state in statesToAdd:
                self.possibleStates.add(state)
            #if statesToAdd and tran.reward != -1:
            #    self.scoreUse += 1 
            if self.efsm.isFSM and self.possibleStates:
                possibleFSMs.append(self.id)
                #initialReached = (len(self.possibleStates) == 1 and 0 in self.possibleStates)  
        return initialReached
    
class GrammarLearner(BaseLearner):
    def __init__(self):
        self.trace = []
        self.prevTrace = []
        self.alphabet = string.ascii_letters + string.digits + ' ,.!;?-'
        self.X = set(self.alphabet)
        self.G = EFSMhierarchyNode(None, EFSM(), 0) # root of the hierarchy
        self.G.efsm.addTransition(0, Transition(LABEL_FORALL,self.alphabet), 0)
        self.FSMid = 0
        self.logger = logging.getLogger(__name__)
        #self.logger.info("X {0}".format(self.X))
        self.file = open("efsm.txt", 'w')
        self.idx = 0
        self.outputsToTry = set(self.X)
        self.startLearning()
        self.G.possibleStates.clear() # the root has no child and no knowledge at first

    def log(self, msg, con=False):
        if con:
            self.logger.info(msg)
        print(msg, file=self.file)
    
    def startLearning(self):
        self.learning = True
        self.consecutiveRewards = 0
        self.G.initPossiblePossitions(self.trace)
        self.hypothesis = EFSM(True)
        for tran in self.trace:
            self.hypothesis.addTransition(0, tran, 0)
            if tran.reward != 1:
                self.outputsToTry.discard(tran.output)


    def reward(self, reward):
        if not self.trace: # the very first reward (for nothing)
            return
        self.trace[-1].reward = reward
        glog("{2} last transition {0} - {1}".format(self.trace[-1], self.learning, self.idx), self)
        self.idx += 1
        initialStateReached = False
        if not self.learning:
            self.learning, initialStateReached = self.G.updateCurrentPosition(self.FSMid, self.trace[-1])
            if self.learning:
                if self.consecutiveRewards > MIN_CONSECUTIVE_REWARDS: # task instance probably learned
                    # wrap the learnt FSM
                    # TODO
                    self.prevTrace.clear()
                    self.outputsToTry = set(self.X)
                self.startLearning()
                self.G.removeLastAddedTransitions(self.FSMid)
            else:
                if reward == 1:
                    self.consecutiveRewards += 1
        else:
            self.hypothesis.addTransition(0, self.trace[-1], 0)
        if self.learning:
            possibleFSMs = []
            initialStateReached = self.G.updatePossiblePositions(self.trace, possibleFSMs)
            if len(possibleFSMs) == 1:
                self.FSMid = possibleFSMs[0]
            if reward == 1: #TODO better condition
                self.G.updateWithNewEFSM(self.hypothesis, self.G.id + 1)
                self.learning, initialStateReached = False, True    
            glog("G: {0}".format(self.G), self)                
        if initialStateReached:
            self.prevTrace += self.trace
            self.trace.clear()
            glog("Hyp: {0}\nPrevTrace: {1}\nTrace{2}".format(self.hypothesis, self.prevTrace, self.trace), self)
            
    def guess(self, x, negOut):
        if self.outputsToTry:
            return random.sample(self.outputsToTry, 1)[0]
        #negOut = self.G.getNegativeOutputs(x)
        notApplied = self.X - negOut
        #self.logger.info("{0} -> {1}".format(negOut,notApplied))
        if notApplied:
            return random.sample(notApplied, 1)[0]
        return random.sample(self.X, 1)[0]
        
    def next(self, x):
        if not self.learning:
            output = self.G.getOutput(self.FSMid, x)
            if not output:
                self.startLearning()
        if self.learning:
            cnt = Counter()
            if self.G.getEstimatedOutputs(x, self.FSMid, self.trace, cnt) \
                and cnt.most_common(1)[0][1] > 0:
                output = cnt.most_common(1)[0][0]
            else:
                cntMP = Counter()
                self.G.getMostProbableOutputs(x, self.trace, cntMP)
                for t in self.trace:
                    if t.input == x:
                        if t.reward == -1:
                            cntMP[t.output] = -2
                        else:
                            cntMP[t.output] += 2
                glog("L Counts: {0}\nMP: {1}".format(cnt, cntMP))
                for k,v in cnt:
                    cntMP[k] += v
                if cntMP:
                    if cntMP.most_common(1)[0][1] > 0:
                        glog("Most probable: {0}".format(cntMP.most_common(1)))
                        output = cntMP.most_common(1)[0][0]
                    else:
                        output = self.guess(x, cnt.keys())
                else:
                    output = self.guess(x, self.hypothesis.getNegativeRewardOutputs(0, x))
            self.outputsToTry.discard(output)
        tran = Transition(x,output)
        self.trace.append(tran)
        return output
