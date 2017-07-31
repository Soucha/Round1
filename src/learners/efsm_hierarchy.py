# Copyright (c) Michal Soucha

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
from learners.efsm import *

class EFSMhierarchyNode:
    def __init__(self, parent, fsm, id, estimated=True):
        self.parent = parent
        self.children = []
        self.id = id
        self.estimated = estimated
        self.scoreUse = 0
        self.efsm = fsm
        self.currState = 0
        self.possibleStates = set()
        self.expectedReward = REWARD_ANY
        #self.logger = logging.getLogger(__name__)

    def __str__(self):
        desc = "use:{3},id:{4} State {0} {5} of\n{1} with {2} children:\n".format(self.currState, self.efsm,
                                                                       len(self.children), self.scoreUse, self.id, self.possibleStates)
        for child in self.children:
            desc += "=> {0}\n".format(child)
        return desc

    def __repr__(self):
        return self.__str__()
    
    def writeDOT(self, dotFile):
        print(' subgraph {', file=dotFile)
        if (self.estimated):
            print('  node [color=green,fillcolor=lightgreen];', file=dotFile)
        self.efsm.writeDOT(dotFile, self.id)
        print(' }', file=dotFile)
        if self.children:
            rank = '{ rank=same;'
            for child in self.children:
                child.writeDOT(dotFile)
                print(' g{0} -> g{1} [color=lightgray,label="{2}",style=dashed];'.format(self.id, child.id, child.scoreUse), file=dotFile)
                rank += ' g{0}'.format(child.id)
            rank += '}'
            print(rank, file=dotFile)

## output estimation
    def getActualTransition(self, tran, checkGuards=True):
        return self.efsm.getTransition(self.currState, tran.input, tran.output, tran.reward, checkGuards)
    
    def getMostRelevantTransition(self, x):
        ## returns the most specified transition for x according to the followed grammar
        tran = self.efsm.getTransition(self.currState, x, None, REWARD_NONNEGATIVE)
        if not tran and self.parent:
            tran, node = self.parent.getMostRelevantTransition(x)
        else:
            node = self
        return tran, node

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
                        outputs = self.efsm.getOutputs(state, REWARD_NONNEGATIVE)
                        if outputs:
                            negOuts = self.efsm.getOutputs(state, REWARD_NEGATIVE, x)
                            for o in outputs.difference(negOuts):
                                counts[o] += 1 # self.scoreUse
                                estimated = True;
        return estimated
    
## update

    def initPossibleStates(self, trace):
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
                child.initPossibleStates(trace)

    def updatePossibleStates(self, tran):
        consistentEFSMs = []
        if self.possibleStates:
            for child in self.children:
                consistentEFSMs.extend(child.updatePossibleStates(tran))
                
            statesToRemove = []
            statesToAdd = []
            for state in self.possibleStates:
                nextState = self.efsm.getNextState(state, tran)
                #glog("upp {0}-{1}->{2} ".format(state, tran, nextState))
                if nextState != NULL_STATE: # comparison with None as nextState could be 0 -> False
                    #if state != 0:
                    statesToRemove.append(state)
                    statesToAdd.append(nextState)
                    if tran.reward != -1:
                        self.scoreUse += 1
                else:
                    t = self.efsm.getTransition(state, tran.input)
                    if t and t.reward is not None:
                        if t.reward != REWARD_NEGATIVE:
                            if t.output != LABEL_EXISTS and ((t.reward != tran.reward and tran.output in t.output) \
                                or (tran.reward != REWARD_NEGATIVE and not tran.output in t.output)):
                                statesToRemove.append(state)
                                if tran.output in t.output:
                                    self.scoreUse -= 1
                                continue
                            elif t.output == LABEL_EXISTS and not consistentEFSMs:
                                statesToRemove.append(state)
                                self.scoreUse -= len(self.children)
                                continue
                        if tran.reward != REWARD_NEGATIVE:
                            t = self.efsm.getTransition(state, tran.input, tran.output, REWARD_NEGATIE)
                            if t:
                                statesToRemove.append(state)
                                self.scoreUse -= 1
            for state in statesToRemove:
                self.possibleStates.discard(state)
            for state in statesToAdd:
                self.possibleStates.add(state)
            #if statesToAdd and tran.reward != -1:
            #    self.scoreUse += 1 
            if not self.possibleStates:
                consistentEFSMs = []
            elif not consistentEFSMs:
                consistentEFSMs.append(self)
        return consistentEFSMs

    def reset(self):
        if not self.efsm.isFSM:
            self.currState = 0
            self.possibleStates = set()
            self.efsm.reset()
            self.expectedReward = REWARD_ANY
            for child in self.children:
                child.reset()
  
    def getConsistentNodes(self, tran):
        consistentNodes = []
        if self.currState != NULL_STATE and not self.estimated:
            for child in self.children:
                consistentNodes.extend(child.getConsistentNodes(tran))
            
            self.scoreUse += tran.reward
            t = self.efsm.getTransition(self.currState, tran.input, tran.output, tran.reward)
            if not t and self.id == 0 and self.currState == 1:
                if not consistentNodes:
                    consistentNodes.append(self)
            elif t and compareRewards(self.expectedReward, tran.reward):
                self.currState = self.efsm.getNextState(self.currState, t)
                if self.currState == 0:
                    self.expectedReward = REWARD_ANY
                if t.action:
                    output = self.efsm.processAction(t, tran.input)
                    if output and output != SYMBOL_UNKNOWN:
                        if output == tran.output:
                            self.expectedReward = REWARD_NONNEGATIVE
                        else:
                            self.expectedReward = REWARD_NONPOSITIVE
                if not consistentNodes:
                    consistentNodes.append(self)
            else:
                self.currState = NULL_STATE
                consistentNodes = []
        return consistentNodes

    def updateCurrentState(self, tran):
        t = self.efsm.getTransition(self.currState, tran.input, tran.output)
        if t:
            if t.reward > REWARD_POSITIVE or t.reward == tran.reward:
                self.currState = self.efsm.getNextState(self.currState, t)
        # TODO else?
            if tran.reward != REWARD_NEGATIVE:
                self.scoreUse += 1
            #else:
             #   self.scoreUse -= 1
        if self.parent:
            self.parent.updateCurrentState(tran)

    def moveAlong(self, tran, findTranAndProcess=True):
        output = None
        if findTranAndProcess:
            x = tran.input
            t = self.efsm.getTransition(self.currState, tran.input, tran.output, tran.reward)
            if not t:
                if tran.input == ' ':
                    t = self.efsm.getTransition(self.currState, tran.input, tran.output, tran.reward, False)
                else:
                    return output
            elif t.action:
                output = self.efsm.processAction(t, x)
            tran = t
        self.currState = self.efsm.getNextState(self.currState, tran)
        return output    

    def updateTransition(self, estimatedStartState, input, newAction=None, newGuard=None):
        state, tran, nextState = self.efsm.findTransition(estimatedStartState, input)
        self.efsm.updateTransition(state, tran, None, newAction, newGuard)

    def updateTransitionsToState(self, targetState, input, newAction=None, newGuard=None):
        trans = self.efsm.findTransitionsToState(targetState, input)
        for state, tran in trans:
            self.efsm.updateTransition(state, tran, None, newAction, newGuard)

    def updateIdByEstimated(self, id):
        if self.estimated:
            self.id = id
            return self.parent.updateIdByEstimated(id + 1)
        return id - 1

    def updateNodePosition(self, H, X):
        if not self.efsm.isFSM and H.efsm.isSpecializationOf(self.efsm, X, H.estimated):
            self.children.append(H)
            H.parent = self
            maxId = self.updateIdByEstimated(H.id + 1)
            return H.tryGeneralize(maxId)
        else:
            if self.estimated:
                self.parent.children.remove(self)
                self.parent.children.extend(self.children)
            return self.parent.updateNodePosition(H, X)

    def makePermanent(self):
        if self.estimated:
            self.estimated = False
            return max(self.id, self.parent.makePermanent())
        return self.id

    def isSpecializationOfParent(self, X=set()):

        return self.efsm.isSpecializationOf(self.parent.efsm, X, self.estimated)

## design of generalized nodes

    def tryGeneralize(self, maxId):
        genEFSMs = self.efsm.tryGeneralize(self.parent.efsm)
        if genEFSMs:
            maxId = self.parent.updateWithGeneralizedEFSMs(genEFSMs, self, maxId)
        else:
            maxId = self.parent.tryGeneralizeChildren(self, maxId)
        return maxId

    def tryGeneralizeChildren(self, relatedChild, maxId):
        for child in reversed(self.children):
            if relatedChild != child:
                genEFSMs = relatedChild.efsm.tryGeneralizeWith(child.efsm, self.efsm)
                if genEFSMs:
                    maxId = self.updateWithGeneralizedEFSMs(genEFSMs, relatedChild, maxId)
                    return self.tryGeneralizeChildren(self.children[-1], maxId)
        return maxId

    def updateWithGeneralizedEFSMs(self, genEFSMs, relatedChild, maxId):
        generalizedEFSM = genEFSMs.pop()
        maxId += 1
        hierNode = EFSMhierarchyNode(self, generalizedEFSM, maxId, False)
        updatedChildren = []
        for child in self.children:
            if child.efsm.isSpecializationOf(generalizedEFSM):
                hierNode.children.append(child)
                child.parent = hierNode
                hierNode.scoreUse += child.scoreUse
                if child.estimated:
                    hierNode.estimated = True
            else:
                updatedChildren.append(child)
        updatedChildren.append(hierNode)
        self.children[:] = updatedChildren
        if genEFSMs:
            maxId = hierNode.updateWithGeneralizedEFSMs(genEFSMs, relatedChild, maxId)
        else:
            maxId = hierNode.tryGeneralizeChildren(relatedChild, maxId)
        return self.tryGeneralizeChildren(hierNode, maxId)

    def removeLastAddedTransitions(self, trace):
        state = 0
        for tran in trace:
            ns = self.efsm.getNextState(state, tran)
            self.efsm.removeTransition(state, tran)
            state = ns

## not in use

    def getOutput(self, FSMid, x):
        ## returns (guessed) output according to the followed grammar
        output = self.efsm.getNonnegativeRewardOutput(self.currState, x)
        if output:
            if len(output) > 1: # output is not a particular symbol
                output = self.children[-1].getOutput(FSMid, x)
        return output

    
    def updateCurrentPositionDown(self, FSMid, tran, parentTran=None):
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

   
    
    def getEstimatedOutputs(self, x, trace, counts):
        ## 
        estimated = False
        if self.possibleStates:
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
                                        if chState != NULL_STATE:
                                            chTran = child.efsm.getTransition(chState, x, None, t.reward)
                                            if chTran and len(chTran.input) == len(chTran.output) == 1: # not a mapping
                                                if chTran.input == x:
                                                    counts[chTran.output] += 1
                                                else:
                                                    counts[chTran.output] -= 1
                                            for chOut in child.efsm.getOutputs(chState, x, REWARD_NEGATIVE):
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
                                        if chState != NULL_STATE:
                                            chTran = child.efsm.getTransition(chState, x, None, t.reward)
                                            if chTran:
                                                counts[chTran.output] += 1
                                            for chOut in child.efsm.getOutputs(chState, x, REWARD_NEGATIVE):
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

    def getEstimatedOutputs(self, x, FSMid, trace, counts):
        ## 
        estimated = False
        if self.possibleStates:
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
                                        if chState != NULL_STATE:
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
                                        if chState != NULL_STATE:
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

        
    def getNegativeOutputs(self, x):
        negOuts = set()
        for state in self.possibleStates:
            negOuts |= self.efsm.getOutputs(state, x, REWARD_NEGATIVE)
        for child in self.children:
            negOuts |= child.getNegativeOutputs(x)
        return negOuts
    
    
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
                if nextState != NULL_STATE: # comparison with None as nextState could be 0 -> False
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
