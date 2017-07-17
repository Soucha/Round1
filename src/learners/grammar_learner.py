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
    
    def writeDOT(self):
        dotFile = open("kb.gv", 'w')
        print('digraph { { rank=min; g0s0 }', file=dotFile)
        self.G.writeDOT(dotFile)
        print('}', file=dotFile)

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
                self.FSMid += 1
                self.G.updateWithNewEFSM(self.hypothesis, self.FSMid)
                self.writeDOT()
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
