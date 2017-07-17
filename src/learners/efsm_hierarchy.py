# Copyright (c) Michal Soucha

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
from learners.efsm import *

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
    
    def writeDOT(self, dotFile):
        print(' subgraph g{0} {{'.format(self.id), file=dotFile)
        self.efsm.writeDOT(dotFile, self.id)
        print(' }', file=dotFile)
        if self.children:
            rank = '{ rank=same;'
            for child in self.children:
                child.writeDOT(dotFile)
                print(' g{0}s0 -> g{1}s0 [color=lightgray,label="{2}",style=dashed];'.format(self.id, child.id, self.scoreUse), file=dotFile)
                rank += ' g{0}s0'.format(child.id)
            rank += '}'
            print(rank, file=dotFile)

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
        if self.id >= hierNode.id:
            hierNode.id = self.id
            self.id = hierNode.id + 1
        else:
            hierNode.id += 1
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
                        #if self.id < node.id:
                         #   self.id = node.id
                        return self
                    break
            if not node:
                node = EFSMhierarchyNode(self, hypothesis, newId)
                self.children.append(node)
            #if self.id < node.id:
             #   self.id = node.id
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
            if t and t.reward != tran.reward and tran.output == t.output:
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
