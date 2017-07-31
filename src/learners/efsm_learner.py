# Copyright (c) Michal Soucha

# tasks_config.challenge.json -l learners.efsm_learner.GrammarLearner

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

def getBasicCommunicationEFSM():
    efsm = EFSM()
    efsm.addTransition(0, Transition(' ',' ', REWARD_ANY, getActionBeginNewWord()), 0)
    efsm.addTransition(0, Transition('.','', REWARD_ANY, getActionSetOutput(SYMBOL_UNKNOWN_STR)), 1)
    efsm.addTransition(0, Transition(LABEL_OTHERS,' ', REWARD_ANY, getActionAppendSymbol()), 0)

    efsm.addTransition(1, Transition(' ','', REWARD_ANY, getActionGetOutputSymbol(), getGuardOnOutput(True)), 1)
    efsm.addTransition(1, Transition(' ','.', REWARD_ANY, getActionBeginNewWord(), getGuardOnOutput(False)), 2)

    efsm.addTransition(2, Transition(' ',' ', REWARD_ANY, getActionBeginNewWord()), 2)
    efsm.addTransition(2, Transition(';',' ', REWARD_ANY, getActionClearWords()), 0)
    efsm.addTransition(2, Transition(LABEL_OTHERS,' ', REWARD_ANY, getActionAppendSymbol()), 2)
    return efsm

class GrammarLearner(BaseLearner):
    def __init__(self):
        self.trace = [[]]
        alphabet = string.ascii_letters + string.digits + ' ,.!;?-'
        self.X = set(alphabet)
        self.maxId = 0
        self.G = EFSMhierarchyNode(None, getBasicCommunicationEFSM(), self.maxId, False) # root of the hierarchy
        self.possibleNodes = [self.G]
        #self.logger = logging.getLogger(__name__)
        self.idx = 0
        self.startLearning(self.G)
        #self.G.updateNodePosition(self.H, self.X)
        #self.G.initPossibleStates(self.trace)
        self.writeDOT()
        
    def writeDOT(self):
        dotFile = open("kb.gv", 'w')
        print('digraph { { rank=min; g0s0 }', file=dotFile)
        self.G.writeDOT(dotFile)
        print('}', file=dotFile)

    def startLearning(self, parentBase=None):
        self.expectedReward = REWARD_ANY
        self.consecutiveRewards = 0
        self.taskInstanceLearnt = False
        self.learning = True
        self.G.reset()
        for tran in self.trace[-1]:
            self.possibleNodes = self.G.getConsistentNodes(tran)
        self.initH(parentBase)
        
    def finishLearning(self, checkParent):
        if not self.trace[-1]:
            self.trace.pop()
        del self.trace[:-1]
        #self.H.removeLastAddedTransitions(self.trace[-1])
        self.H.parent.children.remove(self.H) # removes H as all important information are in H.parent
        self.H.parent.parent.children.remove(self.H.parent)
        if self.H.parent.efsm.mapping:
            self.H.parent.efsm.fixedMapping = True
        if checkParent and self.H.parent.isSpecializationOfParent():
            self.H.parent.estimated = False
            self.maxId = self.H.parent.parent.updateNodePosition(self.H.parent, self.X)
        else:
            self.maxId -= 2 #self.H.parent.makePermanent()
        self.writeDOT()
               
    def guess(self, x, negOuts=None):
        if not negOuts:
            negOuts = self.H.efsm.getOutputs(self.H.currState, REWARD_NEGATIVE)
            if len(negOuts) == len(self.X):
                negOuts = self.H.efsm.getOutputs(self.H.currState, REWARD_NEGATIVE, x)
        notApplied = self.X - negOuts
        if notApplied:
            return random.sample(notApplied, 1)[0]
        return random.sample(self.X, 1)[0]
    
    def getOutput(self, x):
        tran, relNode = self.H.parent.getMostRelevantTransition(x)
        if not tran:
            self.expectedReward = None
            return None
        if relNode == self.H.parent:
            output = self.H.parent.efsm.processAction(tran, x) if tran.action else None
            self.nextTransition = tran
        else:
            output = None
            self.nextTransition = None
            #ns = relNode.efsm.getNextState(relNode.currState, tran)
            #if ns == 0 and relNode.currState != ns:
            #    self.H.parent.currState = 0
        if output and output != SYMBOL_UNKNOWN:
            self.expectedReward = REWARD_NONNEGATIVE
        elif len(tran.output) == 1:
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
        else:
            output = None
            self.expectedReward = REWARD_ANY
        return output

    def next(self, x):
        output = self.getOutput(x) 
        if not output:
            cnt = Counter()
            if cnt:# self.H.getEstimatedOutputs(x, self.maxId, self.trace, cnt) \
                #and cnt.most_common(1)[0][1] > 0:
                output = cnt.most_common(1)[0][0]
            else:
                cntMP = Counter()
                #self.G.getMostProbableOutputs(x, self.trace, cntMP)
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
        self.trace[-1].append(Transition(x,output))
        return output

    def updateHypothesis(self, tran, initialStateReached=False):
        if initialStateReached or (self.H.currState != 0 and self.H.parent.currState == 0):
            ns = 0
            self.trace.append([]) # new communication cycle
        else:
            ns = self.H.efsm.numberOfStates
        retVal = self.H.efsm.addTransition(self.H.currState, tran, ns)
        if retVal != NULL_STATE:
            ns = retVal
        self.H.currState = ns    

    def learnActions(self):
        self.H.parent.updateTransitionsToState(1, '.', getActionOutputMapping(getWord(0)))
        self.H.parent.updateTransitionsToState(0, ';', getActionUpdateMapping(getWord(0), getWord(1)), getGuardOnMapping(getWord(0), getWord(1)))
        self.H.parent.efsm.initMappingBySimulation(self.trace)
        self.learning = False

    def createPlaceForH(self, id, parentBase=None, updateH=False):
        if parentBase and parentBase not in self.possibleNodes:
            parentBase = None
        if not parentBase:
            parentBase = max(self.possibleNodes, key=lambda el: el.scoreUse)
        parent = EFSMhierarchyNode(parentBase, parentBase.efsm.copy(), id)
        parentBase.children.append(parent)
        self.H.parent = parent
        parent.children.append(self.H)
        self.nextTransition = None
        for tran in self.trace[-1]:
            self.H.parent.moveAlong(tran)
            if updateH:
                self.updateHypothesis(tran)

    def initH(self, parentBase):
        self.maxId += 2
        self.H = EFSMhierarchyNode(None, EFSM(True), self.maxId)
        self.createPlaceForH(self.maxId-1, parentBase, True)
        
    def checkH(self):
        if self.H.currState == 0 and self.learning:
            if len(self.H.efsm.transitions[0]) > 2:
                self.H.parent.efsm.trySpecializeWith(self.H.efsm, self.trace)
                self.learnActions()

    def reward(self, reward):
        if not self.trace[0]: # the very first reward (for nothing)
            return
        self.trace[-1][-1].reward = reward
        glog("{0} g{1} last transition {2}".format(self.idx, self.maxId, self.trace[-1][-1]), self)
        self.idx += 1
        updateByTran = True
        if reward == REWARD_POSITIVE:
            self.consecutiveRewards += 1
            if self.consecutiveRewards > MIN_CONSECUTIVE_REWARDS:
                self.taskInstanceLearnt = True
        elif reward == REWARD_NEGATIVE:
            if self.expectedReward != REWARD_ANY:
                self.learning = True
                if self.taskInstanceLearnt:
                    self.finishLearning(True)
                    self.startLearning(self.H.parent)
                    updateByTran = False
            self.consecutiveRewards = 0
        if updateByTran:
            globalState = self.G.currState
            self.possibleNodes = self.G.getConsistentNodes(self.trace[-1][-1])
            if not self.nextTransition or not compareRewards(self.nextTransition.reward, reward):
                self.nextTransition = self.H.parent.getActualTransition(self.trace[-1][-1], False)
            if self.nextTransition:
                self.H.parent.moveAlong(self.nextTransition, False)
            else:
                self.H.parent.currState = NULL_STATE
            self.updateHypothesis(self.trace[-1][-1], globalState != 0 and self.G.currState == 0) # last tran added in startLearning
            
        if self.H.parent.currState != NULL_STATE and self.H.isSpecializationOfParent(self.X) and \
            self.H.parent.parent in self.possibleNodes:
            #self.H.updateCurrentState(self.trace[-1])
            #self.maxId = self.H.tryGeneralize(self.maxId)
            pass
        elif self.taskInstanceLearnt:
        #elif self.consecutiveRewards > MIN_CONSECUTIVE_REWARDS or \
        #    (self.consecutiveRewards > 0 and all(tran.reward != REWARD_POSITIVE for tran in self.trace[-1]) \
        #    and not self.H.parent.isSpecializationOfParent()): # task instance learnt
            self.finishLearning(True) #self.consecutiveRewards > MIN_CONSECUTIVE_REWARDS)
            self.startLearning()
        else:
            self.H.parent.parent.children.remove(self.H.parent)
            self.createPlaceForH(self.H.parent.id)
            self.learning = True
            #self.maxId = self.H.parent.updateNodePosition(self.H, self.X)
            #self.G.updatePossibleStates(self.trace) # update scoreUse
        self.checkH()

        self.writeDOT()
        self
    