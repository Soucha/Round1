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
import pickle
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
    efsm.addTransition(1, Transition(' ','.', REWARD_ANY, "", getGuardOnOutput(False)), 2)
    efsm.addTransition(1, Transition(LABEL_OTHERS,' ', REWARD_ANY, getActionAppendSymbol()), 2) # Teacher stops the learner by speaking

    efsm.addTransition(2, Transition(' ',' ', REWARD_ANY, getActionBeginNewWord()), 2)
    efsm.addTransition(2, Transition(';',' ', REWARD_ANY, getActionClearWords()), 0)
    efsm.addTransition(2, Transition(LABEL_OTHERS,' ', REWARD_ANY, getActionAppendSymbol()), 2)
    return efsm

BRAIN_FN = "brainTmp.pkl"
BRAIN_FNo = ""

class GrammarLearner(BaseLearner):
    def __init__(self, brainFileName=BRAIN_FNo, dotFileName="kb.gv"):
        alphabet = string.ascii_letters + string.digits + ' ,.!;?-'
        self.X = set(alphabet)
        self.dotFN = dotFileName
        self.loadBrain(brainFileName)

        self.idx = 0
        self.trace = [[]]
        self.startLearning(self.possibleNodes[0])
        self.writeDOT()
        
    def writeDOT(self):
        try:
            if self.dotFN:
                with open(self.dotFN, 'w') as dotFile:
                    print('digraph { { rank=min; g0s0 }', file=dotFile)
                    self.G.writeDOT(dotFile)
                    print('}', file=dotFile)
        except:
            print("Unable to write the DOT file!")
        

    def storeBrain(self, lastH):
        try:
            with open(self.brainFN, 'wb') as f:
                pickle.dump((self.maxId, self.G, lastH), f, pickle.HIGHEST_PROTOCOL)
        except:
            print("Unable to store the brain!")
        
    def loadBrain(self, brainFileName):
        try:
            self.possibleNodes = []
            if brainFileName:
                self.brainFN = brainFileName
                with open(self.brainFN, 'rb') as f:
                    (self.maxId, self.G, lastH) = pickle.load(f)
                    self.possibleNodes = [lastH]
        except:
            print("Unable to load the provided brain!")
        finally:
            if not self.possibleNodes:
                self.maxId = 0
                efsm = getBasicCommunicationEFSM() # EFSM()
                #efsm.addTransition(0, Transition(LABEL_OTHERS, '', REWARD_ANY, getActionSetOutput(SYMBOL_UNKNOWN_STR)), 0)
                self.G = EFSMhierarchyNode(None, efsm, self.maxId, False) # root of the hierarchy
                #self.maxId += 1
                #node = EFSMhierarchyNode(self.G, getBasicCommunicationEFSM(), self.maxId, False)
                #self.G.children.append(node)
                self.possibleNodes = [self.G] #node]
                self.brainFN = BRAIN_FN

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
        self.H.parent.estimated = False
        if checkParent and self.H.parent.isSpecializationOfParent():
            self.maxId = self.H.parent.parent.updateNodePosition(self.H.parent, self.X)
            self.storeBrain(self.H.parent)
        else:
            self.maxId -= 2 #self.H.parent.makePermanent()
            self.storeBrain(self.H.parent.parent)
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
            if tran.action:
                output, processed = self.H.parent.efsm.processAction(tran, x)
            else:
                output = None
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
            output = self.guess(x, {'.'}) # None
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

    def updateHypothesis(self, tran, initialStateReached):
        if initialStateReached: # or (self.H.currState != 0 and self.H.parent.currState == 0):
            ns = 0
            self.trace.append([]) # new communication cycle
        else:
            ns = self.H.efsm.numberOfStates
        retVal = self.H.efsm.addTransition(self.H.currState, tran, ns)
        if retVal != NULL_STATE:
            ns = retVal
        self.H.currState = ns    

    def testEstimatedAction(self, pos, neg, action, guard="", \
            mappingAction="", mappingGuard="", mappingActionNeg="", mappingGuardNeg=""):
        efsm = EFSM()
        tran = Transition('','',0, action, guard)
        mappingTran = Transition('','',0, mappingAction, mappingGuard) if mappingAction else None
        for input, outs in pos.items():
            if mappingTran:
                for (output, feedback) in outs:
                    efsm.words = list(input) + list(feedback)
                    y, processed = efsm.processAction(mappingTran, '.')
                    if not processed:
                        return False
            efsm.words = list(input)
            y, processed = efsm.processAction(tran, '.')
            if not y or not processed:
                return False
            expOut = y + ''.join(efsm.output)
            if any([expOut != output for (output, feedback) in outs]):
                return False
        for input, outs in neg.items():
            if mappingTran:
                for (output, feedback) in outs:
                    efsm.words = list(input) + list(feedback)
                    y, processed = efsm.processAction(mappingTran, '.')
                    if not processed:
                        return False
            efsm.words = list(input)
            y, processed = efsm.processAction(tran, '.')
            if not y or not processed:
                return False
            expOut = y + ''.join(efsm.output)
            if any([expOut == output for (output, feedback) in outs]):
                return False
        
        self.H.parent.updateTransitionsToState(1, '.', action, guard)
        if mappingTran:
            self.H.parent.updateTransitionsToState(0, ';', mappingAction, mappingGuard)
            self.H.parent.efsm.initMappingBySimulation(self.trace)
            self.H.parent.efsm.useOfMapping = MAPPING_FIXED 
        self.learning = False
        return True

    def learnActions(self):
        parent = self.H.parent
        pos = {}
        neg = {}
        for trace in self.trace:
            stage = 1
            hasNegativeReward = False
            for tran in trace:
                if tran.reward == REWARD_NEGATIVE:
                    hasNegativeReward = True
                if stage == 1: # assignment
                    if tran.input == '.':
                        stage = 2
                        output = str(tran.output)
                        input = tuple(parent.efsm.words)
                elif stage == 2: # learner output
                    if tran.output == '.':
                        stage = 3
                    elif tran.input != ' ':
                        stage = 3
                        if tran.output != ' ':
                            output += tran.output
                    else:
                        output += tran.output
                elif stage == 3: # feedback
                    if tran.input == ';':
                        feedback = parent.efsm.words.copy()
                        del feedback[:len(input)]
                        storage = neg if hasNegativeReward else pos
                        if input in storage:
                            storage[input].add((output, tuple(feedback)))
                        else:
                            storage[input] = {(output, tuple(feedback))}
                parent.moveAlong(tran)
        action = ""
        guard = ""
        if pos:
            for input, outs in pos.items():
                negOuts = neg.get(input, set())
                for (output, feedback) in outs:
                    for i in range(len(input)):
                        if input[i] == output:
                            action = getActionSetOutput(getWord(i))
                            guard = getGuardOnWordCount(i + 1)
                            if self.testEstimatedAction(pos, neg, action, guard):
                                return
                    for i in range(len(feedback)):
                        if feedback[i] == output:
                            action = getActionOutputMapping(getWord(0))
                            guard = getGuardOnWordCount(1)
                            mappingAction = getActionUpdateMapping(getWord(0), getWord(1))
                            mappingGuard = connectGuards([getGuardOnWordCount(2), getGuardOnMapping(getWord(0), getWord(1))])
                            if self.testEstimatedAction(pos, neg, action, guard, mappingAction, mappingGuard):
                                return

        else:
            # find the shortest feedback
            (input, outs) = min(neg.items(), key=lambda el: len(min(el[1], key=lambda elOutFeed: len(elOutFeed[1]))))
            (refOutput, refFeedback) = min(outs, key=lambda el: len(el[1]))
            for i in range(len(input)):
                for j in range(len(refFeedback)):
                    if input[i][0] == refFeedback[j][0]:
                        step = refFeedback[j].find(input[i][1], 1) if len(input[i]) > 1 else -1
                        if step != -1 and step * len(input[i]) <= len(refFeedback[j]) and \
                            all([refFeedback[j][k*step] == input[i][k] for k in range(len(input[i]))]):
                            if step == 1: # copy of input
                                if i + 1 == len(input):
                                    action = getActionSetOutput(getWord(i))
                                    guard = getGuardOnWordCount(i + 1)
                                    if self.testEstimatedAction(pos, neg, action, guard):
                                        return
                                endIdx = -1
                                for k in range(i + 1, min(len(input), len(refFeedback))):
                                    if input[k] != refFeedback[k]:
                                        endIdx = k
                                        break
                                if endIdx == -1 or endIdx > i + 1:
                                    action = getActionSetOutput(getFunJoin(getWords(i), ' ')) if endIdx == -1 else \
                                        getActionSetOutput(getFunJoin(getWords(i, endIdx), ' '))
                                    guard = "" if endIdx == -1 else getGuardOnWordCount(endIdx)
                                    if self.testEstimatedAction(pos, neg, action, guard):
                                        return
                                endIdx = -1
                                for k in range(i + 1, min(len(input), i + j + 1)):
                                    if input[k] != refFeedback[j-(k-i)]:
                                        endIdx = k
                                        break
                                if endIdx == -1 or endIdx > i + 1:
                                    action = getActionSetOutput(getFunJoin(getFunReversed(getWords(i)), ' ')) if endIdx == -1 else \
                                        getActionSetOutput(getFunJoin(getFunReversed(getWords(i, endIdx)), ' '))
                                    guard = "" if endIdx == -1 else getGuardOnWordCount(endIdx)
                                    if self.testEstimatedAction(pos, neg, action, guard):
                                        return
                            else: # interleaving
                                pass
                    if input[i][-1] == refFeedback[j][-1]: # check for reversed
                        pass
                    # check for substrings

            # no clue -> try mapping
            (input, outs) = max(neg.items(), key=lambda el: len(el[1]))
            if len(outs) > 1: # several outputs and feedbacks to the same input
                # find common part of feedbacks
                (refOutput, refFeedback) = min(outs, key=lambda el: len(el[1]))
                outputs = set([output for (output, _) in outs])
                feedbacks = set([feedback for (_, feedback) in outs])
                isSingleWord = False
                if len(refFeedback) == 1 and all([len(feedback) == 1 for feedback in feedbacks]): # check letter-wise
                    feedbacks = [feedback[0] for feedback in feedbacks]
                    refFeedback = min(feedbacks, key=len)
                    isSingleWord = True
                for j in range(len(refFeedback)):
                    if all([refFeedback[j] == feedback[j] for feedback in feedbacks]):
                        endIdx = -1
                        for k in range(j + 1, len(refFeedback)):
                            if any([refFeedback[k] != feedback[k] for feedback in feedbacks]):
                                endIdx = k
                                break
                        if endIdx == -1:
                            #if any([len(refFeedback) < len(feedback) for feedback in feedbacks]):
                            # could be set to variable end like [j:] instead of [j:endIdx]    
                            endIdx = len(refFeedback)
                        out = refFeedback[j:endIdx] if isSingleWord else ' '.join(refFeedback[j:endIdx])
                        while j < endIdx and out in outputs:
                            endIdx -= 1
                            out = refFeedback[j:endIdx] if isSingleWord else ' '.join(refFeedback[j:endIdx])
                        if j < endIdx:
                            mapKey = getFunJoin(getWords(None, len(input)), ' ')
                            mapVal = getWord(len(input), j, endIdx) if isSingleWord else \
                                getFunJoin(getWords(len(input)+j, len(input)+endIdx), ' ')
                            action = getActionOutputMapping(mapKey)
                            guard = ""
                            mappingAction = getActionUpdateMapping(mapKey, mapVal)
                            mappingGuard = getGuardOnWordLength(len(input), endIdx) if isSingleWord else \
                                getGuardOnWordCount(len(input)+endIdx)
                            mappingGuard = connectGuards([mappingGuard, getGuardOnMapping(mapKey, mapVal)])
                            if self.testEstimatedAction(pos, neg, action, guard, mappingAction, mappingGuard):
                                return
                    if all([refFeedback[-j-1] == feedback[-j-1] for feedback in feedbacks]):
                        endIdx = -1
                        for k in range(j + 1, len(refFeedback)):
                            if any([refFeedback[-k-1] != feedback[-k-1] for feedback in feedbacks]):
                                endIdx = k
                                break
                        if endIdx == -1:
                            #if any([len(refFeedback) < len(feedback) for feedback in feedbacks]):
                            # could be set to variable end like [j:] instead of [j:endIdx]    
                            endIdx = len(refFeedback)
                        out = refFeedback[-endIdx:len(refFeedback)-j] if isSingleWord else ' '.join(refFeedback[-endIdx:len(refFeedback)-j])
                        while j < endIdx and out in outputs:
                            endIdx -= 1
                            out = refFeedback[-endIdx:len(refFeedback)-j] if isSingleWord \
                                else ' '.join(refFeedback[-endIdx:len(refFeedback)-j])
                        if j < endIdx:
                            mapKey = getFunJoin(getWords(None, len(input)), ' ')
                            if j == 0:
                                mapVal = getWord(len(input), -endIdx) if isSingleWord else \
                                    getFunJoin(getWords(-endIdx), ' ')
                            else:
                                mapVal = getWord(len(input), -endIdx, -j) if isSingleWord else \
                                    getFunJoin(getWords(-endIdx, -j), ' ')
                            action = getActionOutputMapping(mapKey)
                            guard = ""
                            mappingAction = getActionUpdateMapping(mapKey, mapVal)
                            mappingGuard = getGuardOnWordLength(len(input), endIdx) if isSingleWord else \
                                getGuardOnWordCount(len(input)+endIdx)
                            mappingGuard = connectGuards([mappingGuard, getGuardOnMapping(mapKey, mapVal)])
                            if self.testEstimatedAction(pos, neg, action, guard, mappingAction, mappingGuard):
                                return                     
            for i in range(len(input)):
                action = getActionOutputMapping(getWord(i))
                guard = getGuardOnWordCount(i + 1)
                for j in range(len(refFeedback)):
                    k = len(input) + j
                    mappingAction = getActionUpdateMapping(getWord(i), getWord(k))
                    mappingGuard = connectGuards([getGuardOnWordCount(k + 1), getGuardOnMapping(getWord(i), getWord(k))])
                    if self.testEstimatedAction(pos, neg, action, guard, mappingAction, mappingGuard):
                        return
                
                
        
        #self.H.parent.updateTransitionsToState(1, '.', getActionOutputMapping(getWord(0)))
        #self.H.parent.updateTransitionsToState(0, ';', getActionUpdateMapping(getWord(0), getWord(1)), \
        #    connectGuards([getGuardOnWordCount(2), getGuardOnMapping(getWord(0), getWord(1))]))
        #self.H.parent.efsm.initMappingBySimulation(self.trace)
        #self.H.parent.efsm.useOfMapping = MAPPING_FIXED
        

    def createPlaceForH(self, id, parentBase=None, updateH=False):
        if parentBase and parentBase not in self.possibleNodes:
            parentBase = None
        if not parentBase:
            parentBase = max(self.possibleNodes, key=lambda el: el.scoreUse)
        parent = EFSMhierarchyNode(parentBase, parentBase.efsm.copy(), id)
        parentBase.children.append(parent)
        self.H.parent = parent
        parent.children.append(self.H)
        self.expectedReward = REWARD_NONNEGATIVE if parent.efsm.useOfMapping else REWARD_ANY
        self.nextTransition = None
        for tran in self.trace[-1]:
            prevState = self.H.parent.currState
            self.H.parent.moveAlong(tran)
            if updateH:
                self.updateHypothesis(tran, prevState != 0 and self.H.parent.currState == 0)

    def initH(self, parentBase):
        self.maxId += 2
        self.H = EFSMhierarchyNode(None, EFSM(True), self.maxId)
        self.createPlaceForH(self.maxId-1, parentBase, True)
        
    def checkH(self):
        if self.H.currState == 0 and self.learning and len(self.H.efsm.transitions[0]) > 2:
            if self.H.parent.efsm.useOfMapping or self.expectedReward == REWARD_NONNEGATIVE:
                self.learning = False
            else:
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
                self.expectedReward = REWARD_ANY
                if self.taskInstanceLearnt:
                    self.finishLearning(True)
                    self.startLearning(self.H.parent)
                    updateByTran = False
            self.consecutiveRewards = 0
        if updateByTran:
            globalState = self.G.currState
            #isInitialState = any([node.currState != 0 for node in self.possibleNodes])
            self.possibleNodes = self.G.getConsistentNodes(self.trace[-1][-1])
            if self.nextTransition and reward == REWARD_NEGATIVE and not compareRewards(self.nextTransition.reward, reward):
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
    