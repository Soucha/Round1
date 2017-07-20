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
from learners.efsm_hierarchy import *

debFile = open("gEfsm.txt", 'w')
def glog(msg, o=None):
    if o:
        print(msg)
    print(msg, file=debFile)

MIN_CONSECUTIVE_REWARDS = 5            
        
class GrammarLearner(BaseLearner):
    def __init__(self):
        self.trace = []
        alphabet = string.ascii_letters + string.digits + ' ,.!;?-'
        self.X = set(alphabet)
        self.maxId = 0
        self.G = EFSMhierarchyNode(None, EFSM(), self.maxId, False) # root of the hierarchy
        self.G.efsm.addTransition(0, Transition(LABEL_FORALL,alphabet), 0)
        
        self.logger = logging.getLogger(__name__)
        self.idx = 0
        self.consecutiveRewards = 0
        self.startLearning()
        self.G.updateNodePosition(self.H, self.X)
        self.G.initPossibleStates(self.trace)
        
    def writeDOT(self):
        dotFile = open("kb.gv", 'w')
        print('digraph { { rank=min; g0s0 }', file=dotFile)
        self.G.writeDOT(dotFile)
        print('}', file=dotFile)

    def startLearning(self):
        self.maxId += 1
        self.H = EFSMhierarchyNode(None, EFSM(True), self.maxId)
        if self.trace:
            for t in self.trace[:-1]:
                self.updateHypothesis(t)
            self.G.initPossibleStates(self.trace[:-1])
            self.H.initPossibleStates(self.trace[:-1])
        
    def guess(self, x, negOuts=None):
        if not negOuts:
            negOuts = self.H.efsm.getOutputs(self.H.currState, REWARD_NEGATIVE)
            if len(negOuts) == len(self.X):
                negOuts = self.H.efsm.getOutputs(self.H.currState, REWARD_NEGATIVE, x)
        notApplied = self.X - negOuts
        if notApplied:
            return random.sample(notApplied, 1)[0]
        return random.sample(self.X, 1)[0]
    
    def getOutputFromTransition(self, x, tran):
        if not tran:
            self.expectedReward = None
            return None
        self.expectedReward = tran.reward
        output = None
        if len(tran.output) == 1:
            output = tran.output
        elif tran.output == LABEL_EXISTS:
            if tran.input == LABEL_FORALL:
                t = self.H.efsm.getTransition(self.H.currState, None, None, tran.reward)
                if t:
                    output = t.output
                else:
                    output = self.guess(x, self.H.efsm.getOutputs(self.H.currState, REWARD_NEGATIVE))
            elif tran.input == LABEL_SUBSETS:
                outputs = self.H.efsm.getOutputs(self.H.currState, tran.reward)
                negOuts = self.H.efsm.getOutputs(self.H.currState, REWARD_NEGATIVE, x)
                outputs -= negOuts
                if outputs:
                    output = random.sample(outputs, 1)[0]
                else:
                    output = self.guess(x, negOuts)
            elif tran.input == LABEL_EXISTS:
                outputs = self.H.efsm.getOutputs(self.H.currState, tran.reward)
                negOuts = self.H.efsm.getOutputs(self.H.currState, REWARD_NEGATIVE, x)
                output = self.guess(x, outputs + negOuts)
        return output

    def next(self, x):
        self.expectedReward = None
        relTran = self.H.getMostRelevantTransition(x)
        output = self.getOutputFromTransition(x, relTran) 
        if not output:
            cnt = Counter()
            if cnt:# self.H.getEstimatedOutputs(x, self.maxId, self.trace, cnt) \
                #and cnt.most_common(1)[0][1] > 0:
                output = cnt.most_common(1)[0][0]
            else:
                cntMP = Counter()
                self.G.getMostProbableOutputs(x, self.trace, cntMP)
                #for t in self.trace:
                #    if t.input == x:
                #        if t.reward == -1:
                #            cntMP[t.output] = -2
                #        else:
                #            cntMP[t.output] += 2
                #glog("L Counts: {0}\nMP: {1}".format(cnt, cntMP))
                #for k,v in cnt:
                #    cntMP[k] += v
                if cntMP:
                    if cntMP.most_common(1)[0][1] > 0:
                        glog("Most probable: {0}".format(cntMP.most_common(1)))
                        output = cntMP.most_common(1)[0][0]
                    else:
                        output = self.guess(x, cnt.keys())
                else:
                    output = self.guess(x)
        self.trace.append(Transition(x,output))
        return output

    def updateHypothesis(self, tran):
        self.H.efsm.addTransition(0, tran, 0)
        #if tran.reward != 1:
         #   self.outputsToTry.discard(tran.output)

    def reward(self, reward):
        if not self.trace: # the very first reward (for nothing)
            return
        self.trace[-1].reward = reward
        glog("{0} g{1} last transition {2}".format(self.idx, self.maxId, self.trace[-1]), self)
        self.idx += 1
        if self.expectedReward:
            if self.expectedReward != reward:
                if self.consecutiveRewards > MIN_CONSECUTIVE_REWARDS: # task instance learnt
                    if self.H.currState != 0:
                        # erase last transitions from H
                        self.trace # TODO update
                        self.H.removeLastAddedTransitions() # from the last initial state
                    else:
                        del self.trace[:-1]
                    self.maxId = self.H.makePermanent()
                    self.writeDOT()
                    self.startLearning()
                #elif was it simulation of a learnt grammar?:
                    #TODO
                self.consecutiveRewards = 0
            elif reward == 1:
                self.consecutiveRewards += 1
        elif reward == 1:
            self.consecutiveRewards += 1
        else:
            self.consecutiveRewards = 0
                
        self.updateHypothesis(self.trace[-1]) # last tran added in startLearning
        if self.H.parent:
            if self.H.efsm.isSpecializationOf(self.H.parent.efsm, self.X):
                self.H.updateCurrentState(self.trace[-1])
                self.maxId = self.H.tryGeneralize(self.maxId)
            else:
                self.H.parent.children.remove(self.H)
                self.maxId = self.H.parent.updateNodePosition(self.H, self.X)
                self.G.updatePossibleStates(self.trace) # update scoreUse
        else:
            possibleEFSMs = self.G.updatePossibleStates(self.trace)
            if len(possibleEFSMs) == 1:
                self.maxId = possibleEFSMs[0].updateNodePosition(self.H, self.X)
        self.writeDOT()
        self
    