# Copyright (c) Michal Soucha

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import Counter

# labels of inputs and outputs - need to contain a letter twice to distinguish from a mapping (=set of letters as a string)
LABEL_FORALL = "forall" # for any input/output
LABEL_EXISTS = "!!1" # there is exactly one
LABEL_SUBSETS = "subsets" # several groups of inputs
LABEL_OTHERS = "else" # all other inputs
LABELS = [LABEL_FORALL, LABEL_SUBSETS, LABEL_EXISTS, LABEL_OTHERS]
VARIABLE = "??" # prefix of a variable

SYMBOL_UNKNOWN = "§"
SYMBOL_UNKNOWN_STR = "'" + SYMBOL_UNKNOWN + "'"

NULL_STATE = -1

MAPPING_NO = 0
MAPPING_USED = 1
MAPPING_FIXED = 2

REWARD_NONPOSITIVE = -2
REWARD_NEGATIVE = -1
REWARD_NEUTRAL = 0
REWARD_POSITIVE = 1
REWARD_NONNEGATIVE = 2
REWARD_ANY = 3

## Returns: 0 if the given rewards are different
##          1 if the first given is more general than the second
##          2 if the second given is more general than the first
##          3 if the given rewards are the same
def compareRewards(r1, r2):
    if r1 == r2:
        return 3
    if r1 == REWARD_ANY:
        return 1
    if r2 == REWARD_ANY:
        return 2
    if r1 == REWARD_NONNEGATIVE:
        if r2 == REWARD_NEUTRAL or r2 == REWARD_POSITIVE:
            return 1
        else:
            return 0
    if r2 == REWARD_NONNEGATIVE:
        if r1 == REWARD_NEUTRAL or r1 == REWARD_POSITIVE:
            return 2
        else:
            return 0
    if r1 == REWARD_NONPOSITIVE:
        if r2 == REWARD_NEUTRAL or r2 == REWARD_NEGATIVE:
            return 1
        else:
            return 0
    if r2 == REWARD_NONPOSITIVE:
        if r1 == REWARD_NEUTRAL or r1 == REWARD_NEGATIVE:
            return 2
        else:
            return 0
    return 0 # r1, r2 are different REWARD_NEGATIVE, REWARD_NEUTRAL, or REWARD_POSITIVE

def getActionBeginNewWord():
    return "self.words[len(self.words):len(self.words)] = [''] if self.words[-1] else []"

def getActionAppendSymbol():
    return "self.words[-1] += x"

def getActionClearWords():
    return "self.words = ['']"

def getActionGetOutputSymbol():
    return "self.y = self.output.pop(0)"

def getActionSetOutput(output):
    return "self.output = list(" + output + "); " + getActionGetOutputSymbol() + "; " + getActionBeginNewWord()

def getActionUpdateMapping(k, v):
    return "self.mapping[" + k + "] = " + v + "; " + getActionClearWords()

def getActionOutputMapping(k):
    return getActionSetOutput("self.mapping.get(" + k + ", " + SYMBOL_UNKNOWN_STR + ")")

def getGuardOnOutput(requireOutput):
    if requireOutput:
        return "self.output"
    return "not self.output"

def getGuardOnWordCount(minCount):
    return "len(self.words) >= " + str(minCount)

def getGuardOnWordLength(wordIdx, minLen):
    return "len(" + getWord(wordIdx) + ") >= " + str(minLen)

def getGuardOnMapping(key, value):
    return key + " not in self.mapping or self.mapping[" + key + "] == " + value

def getGuardOnProhibitedInputs(notAllowedInputs):
    return "x not in '" + notAllowedInputs + "'"

def getProhibitedInputsFromGuard(guard):
    if len(guard) > 10:
        return guard[10:-1]
    return ""

def connectGuards(guards, conjunction="and"):
    if not guards: 
        return ""
    s = "(" + guards[0] + ")"
    for g in guards[1:]:
        s += " " + conjunction + " (" + g + ")"
    return s

def getFunJoin(words, separator=""):
    return "'" + separator + "'.join(" + words + ")"

def getFunReversed(words):
    return "reversed(" + words + ")"

def getWords(startIdx=None, endIdx=None, wordStartIdx=None, wordEndIdx=None):
    w = "self.words"
    if startIdx is not None or endIdx is not None or wordStartIdx is not None or wordEndIdx is not None:
        w += "["
        if startIdx is not None: w += str(startIdx)
        if startIdx != endIdx: w += ":"
        if endIdx is not None and startIdx != endIdx: w += str(endIdx)
        w += "]"
    if wordStartIdx is not None or wordEndIdx is not None:
        w += "["
        if wordStartIdx is not None: w += str(wordStartIdx)
        if wordStartIdx != wordEndIdx: w += ":"
        if wordEndIdx is not None and wordStartIdx != wordEndIdx: w += str(wordEndIdx)
        w += "]"
    return w

def getWord(i, wordStartIdx=None, wordEndIdx=None):
    return getWords(i, i, wordStartIdx, wordEndIdx)


class Transition:
    def __init__(self, x, y, reward=REWARD_ANY, action="", guard=""):
        self.input = x
        self.output = y
        self.reward = reward
        self.action = action
        self.guard = guard

    def __str__(self):
        extStr = ""
        if self.guard:
            extStr += " {0}".format(self.guard)
        if self.action:
            extStr += "\n{0}".format(self.action)
        return "{0}/{1} {2}{3}".format(self.input,self.output,self.reward, extStr)

    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other): 
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash((self.input, self.output, self.reward))

    def copy(self):
        return Transition(str(self.input), str(self.output), int(self.reward), str(self.action), str(self.guard))

    def isSpecializationOf(self, other):
        if self.input == other.input and self.output == other.output and self.reward == other.reward:
            return False
        if compareRewards(self.reward, other.reward) < 2:
            return False
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
        if other.input == LABEL_OTHERS:
            return other.output == self.output
        if not other.output or not self.output:
            return self.input == other.input
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
        self.numberOfStates = 1
        self.transitions = {}
        self.isFSM = isFSM
        self.mapping = {}
        self.useOfMapping = MAPPING_NO
        self.reset()

    def __str__(self):
        desc = "{0}-state ".format(self.numberOfStates)
        if self.isFSM:
            desc += "FSM\n"
        else:
            desc += "EFSM\n"
        if self.mapping:
            desc += "{0}".format(self.mapping)
        #for state in self.transitions:
        #    negDesc = ""
        #    for tran in self.transitions[state]:
        #        if tran.reward != REWARD_NEGATIVE:
        #            desc += "{0} -{1}-> {2}\n".format(state, tran, self.transitions[state][tran])
        #        else:
        #            negDesc += " {0}".format(tran)
        #    if negDesc:
        #        desc += "{0} neg:".format(state) + negDesc + "\n"       
        return desc

    def __repr__(self):
        return self.__str__()

    def writeDOT(self, dotFile, id):
        mappingStr = ""
        for k in sorted(self.mapping.keys()):
            mappingStr += "\n{0} -> {1}".format(k, self.mapping[k])
        print('  g{0} [color=gray,fillcolor=lightgray,label="G{0}{1}",shape=box];'.format(id, mappingStr), file=dotFile)
        print('  g{0} -> g{0}s0 [color=gray];'.format(id), file=dotFile)
        for state in self.transitions:
            for t, ns in self.transitions[state].items():
                print('  g{0}s{1} -> g{0}s{2} [label="{3}"];'.format(id, state, ns, t), file=dotFile)
       
    def reset(self):
        self.words = ['']
        self.output = ""
        self.y = ''
        if self.useOfMapping != MAPPING_FIXED:
            self.mapping = {}
              
    def copy(self):
        efsm = EFSM(self.isFSM)
        efsm.numberOfStates = self.numberOfStates
        if self.mapping:
            efsm.mapping = self.mapping.copy()
        if self.useOfMapping != MAPPING_NO:
            efsm.useOfMapping = MAPPING_FIXED
        for state in self.transitions:
            efsm.transitions[state] = {}
            for tran, ns in self.transitions[state].items():
                efsm.transitions[state][tran.copy()] = ns
        return efsm
           
    def addTransition(self, state, transition, nextState):
        if state in self.transitions:
            if transition in self.transitions[state]:
                return self.transitions[state][transition]
            else:
                self.transitions[state][transition] = nextState
        else:
            self.transitions[state] = {transition: nextState}
        if self.numberOfStates <= max(state, nextState):
            self.numberOfStates = max(state, nextState) + 1
        return NULL_STATE

    def removeTransition(self, state, transition):
        nextState = NULL_STATE
        if state in self.transitions:
            if transition in self.transitions[state]:
                nextState = self.transitions[state].pop(transition)
                if not self.transitions[state]:
                    self.transitions.pop(state)
        return nextState

    def updateTransition(self, state, transition, newReward=None, newAction=None, newGuard=None):
        if state in self.transitions and transition in self.transitions[state]:
            nextState = self.transitions[state].pop(transition)
            if newAction is not None:
                transition.action = newAction
            if newGuard is not None:
                transition.guard = newGuard
            if newReward is not None:
                transition.reward = newReward
            self.transitions[state][transition] = nextState

    def getTransition(self, state, x, y=None, reward=REWARD_ANY, checkGuards=True):
        if state in self.transitions:
            tmpTran = None
            inputNotFound = True
            for tran in self.transitions.get(state):
                if (not x or x == tran.input or tran.input in LABELS or x in tran.input) and \
                    (not y or not tran.output or y == tran.output or tran.output == LABEL_EXISTS or y == LABEL_EXISTS or y in tran.output) \
                    and compareRewards(reward, tran.reward):
                    if x == tran.input:
                        if y == tran.output:
                            if not checkGuards or not tran.guard or eval(tran.guard):
                                return tran
                            else:
                                return None
                        if not checkGuards or not tran.guard or eval(tran.guard) or \
                            (x == ' ' and  not tran.output and (not tmpTran or tmpTran.input != ' ')):
                            tmpTran = tran
                            inputNotFound = True
                        elif not tmpTran or (tmpTran.input != x):
                            inputNotFound = False
                    elif tran.input == LABEL_OTHERS and (not checkGuards or not tran.guard or eval(tran.guard)) and not tmpTran:
                        tmpTran = tran
            if tmpTran and inputNotFound:
                return tmpTran
        return None
    
    def findTransitionsToState(self, targetState, x):
        trans = []
        for state in self.transitions:
            for tran, ns in self.transitions[state].items():
                if ns == targetState and tran.input == x:
                    trans.append((state, tran))
        return trans

    def findTransition(self, estimatedStartState, input):
        statesToCheck = [estimatedStartState]
        i = 0
        while i < len(statesToCheck):
            state = statesToCheck[i]
            if state in self.transitions:
                for tran, ns in self.transitions[state].items():
                    if tran.input == input:
                        return state, tran, ns
                    if ns not in statesToCheck:
                        statesToCheck.append(ns)
            i += 1
    
    def processAction(self, tran, x):
        self.y = None
        if not tran.guard or eval(tran.guard):
            exec(tran.action)
        elif tran.guard:
            return None, False
        return self.y, True
    
    def getOutputs(self, state, reward, x=None):
        outputs = set()
        if state in self.transitions:
            for tran in self.transitions.get(state):
               if (not x or x == tran.input or tran.input in LABELS or x in tran.input) and \
                    compareRewards(reward, tran.reward) and tran.output not in LABELS:
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
        if state in self.transitions:
            if tran in self.transitions[state]:
                return self.transitions.get(state).get(tran)
            t = self.getTransition(state, tran.input, tran.output, tran.reward)
            if t:
                return self.transitions.get(state).get(t)
        return NULL_STATE

    def getEndPathState(self, state, path):
        for input in path:
            tran = self.getTransition(state, input)
            if not tran:
                return NULL_STATE
            nextState = self.getNextState(state, tran)
            if nextState == NULL_STATE:
                return NULL_STATE
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
        if self.useOfMapping == MAPPING_FIXED and self.mapping:
            genEFSM = self.copy()
            genEFSM.useOfMapping = MAPPING_USED
            genEFSM.mapping = {}
            if genEFSM.isSpecializationOf(parentEFSM):
                return [genEFSM]
        return []
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
        return []
        if self.isFSM or other.isFSM:
            return []
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
        return []

    def initMappingBySimulation(self, trace):
        self.reset()
        for sim in trace:
            #if all(tran.reward != REWARD_NEGATIVE for tran in sim):
            state = 0
            self.words = ['']
            for tran in sim:
                t = self.getTransition(state, tran.input, tran.output, tran.reward)
                if not t:
                    t = self.getTransition(state, tran.input, tran.output, tran.reward, False)
                if t.action:
                    output, processed = self.processAction(t, tran.input)
                state = self.getNextState(state, t)

    def long_substr(data):
        substr = ''
        if len(data) > 1 and len(data[0]) > 0:
            for i in range(len(data[0])):
                for j in range(len(data[0])-i+1):
                    if j > len(substr) and all(data[0][i:i+j] in x for x in data):
                        substr = data[0][i:i+j]
        return substr

    def moveTransitions(self, fromState, toState):
        if fromState in self.transitions:
            for tran, ns in self.transitions[fromState].items():
                if ns == fromState:
                    ns = toState
                self.addTransition(toState, tran, ns)
            del self.transitions[fromState]

    def separateState(self, state):
        newState = NULL_STATE
        if state in self.transitions:
            newState = self.numberOfStates
            prohibitedInputs = ""
            selfloops = []
            for tran, ns in self.transitions[state].items():
                if ns == state:
                    ns = newState
                    selfloops.append(tran.copy())
                elif tran.input not in LABELS:
                    prohibitedInputs += tran.input
                self.addTransition(newState, tran, ns)
            del self.transitions[state]
            for tran in selfloops:
                if tran.input == LABEL_OTHERS and prohibitedInputs:
                    tran.guard = getGuardOnProhibitedInputs(getProhibitedInputsFromGuard(tran.guard) + prohibitedInputs)
                self.addTransition(state, tran, state)
        return newState

    def getCommonPrefix(self, transitions):
        prefix = []
        refSeq = min(transitions, key=len)
        for i in range(len(refSeq)):
            if all(refSeq[i].input == seq[i].input and refSeq[i].output == seq[i].output and refSeq[i].reward == seq[i].reward \
                    for seq in transitions):
                prefix.append(refSeq[i])
            else:
                break
        return prefix

    def getCommonSuffix(self, transitions):
        suffix = []
        refSeq = min(transitions, key=len)
        for i in range(-1, -len(refSeq)-1, -1): # last tran in transition is the same as relTran -> len(refSeq)-2
            if all(refSeq[i].input == seq[i].input and refSeq[i].output == seq[i].output and refSeq[i].reward == seq[i].reward \
                    for seq in transitions):
                suffix.append(refSeq[i])
            else:
                break
        return suffix

    def analyseTransitions(self, state, relTran, transitions):
        lastState = state
        #refSeq = min(transitions, key=len)
        # all start with the same prefix
        prefix = self.getCommonPrefix(transitions)
        if prefix:
            lastState = self.separateState(state)
            for tran in prefix[:-1]:
                newState = self.numberOfStates
                self.addTransition(state, tran.copy(), newState)
                state = newState
            self.addTransition(state, prefix[-1].copy(), lastState)
            state = lastState
            for i in range(len(transitions)):
                del transitions[i][:len(prefix)]
            #refSeq = min(transitions, key=len)
        
        # all end with the same suffix
        suffix = self.getCommonSuffix(transitions)
        if suffix:
            targetState = lastState = self.separateState(state)
            for tran in suffix[:-1]: #note that suffix is reversed, i.e. the last tran is the first
                newState = self.numberOfStates
                self.addTransition(newState, tran.copy(), targetState)
                targetState = newState
            self.addTransition(state, suffix[-1].copy(), targetState)
            for i in range(len(transitions)):
                del transitions[i][-len(suffix):]
            #refSeq = min(transitions, key=len)
            
        # all contain the same subsequence

        rewards = set()
        if REWARD_NEGATIVE <= relTran.reward <= REWARD_POSITIVE:
            rewards.add(relTran.reward)
        else:
            for seq in transitions:
                rewards.update([tran.reward for tran in seq])               
        return lastState, rewards
    
    def checkRewards(self, state, tran, nextState, transitions):
        expReward = transitions[0][0].reward
        if all(all(tran.reward == expReward for tran in seq) for seq in transitions) and expReward != tran.reward:
            self.updateTransition(state, tran, expReward)

    def duplicateTransitions(self, stateFrom, stateTo):
        stateMap = {stateFrom : self.numberOfStates, stateTo : stateTo}
        stack = [stateFrom]
        while stack:
            state = stack.pop()
            duplState = stateMap[state]
            for tran, nextState in self.transitions[state].items():
                if nextState in stateMap:
                    ns = stateMap.get(nextState)
                else:
                    ns = self.numberOfStates
                    stateMap[nextState] = ns
                    stack.append(nextState)
                self.addTransition(duplState, tran.copy(), ns) 
        return stateMap

    def updateMappedTraces(self, traces, stateMap):
        for trace in traces:
            if any([tran.reward == REWARD_NEGATIVE for tran in trace[trace[-1]+1]]):
                sIdx = trace[-1]+3
                for i in range(sIdx, len(trace), 3):
                    trace[i] = stateMap[trace[i]]
                    if i > sIdx:
                        trace[i-1] = self.getTransition(trace[i-3], trace[i-1].input, trace[i-1].output, trace[i-1].reward, False)

    def updateStateOrder(self, stateOrder, conditions, stateMap, idx):
        l = len(stateOrder)
        for i in range(idx, l):
            if stateOrder[i] in stateMap:
                newState = stateMap[stateOrder[i]]
                stateOrder.append(newState)
                conditions[newState] = (set(), set(), [], [])
                for k in [2, 3]:    # 2: selfloops state->state, 3: progress transitions
                    for tran in conditions[stateOrder[i]][k]:
                        t = self.getTransition(newState, tran.input, tran.output, tran.reward, False)
                        conditions[newState][k].append(t)

    def addConditions(self, conditions, state, tranOrder, lastTran, nextState):
        if state in conditions:
            conditions[state][1].add(nextState)
            conditions[state][3].add(lastTran)
            if tranOrder:
                if conditions[state][2]:
                    if tranOrder[0] != conditions[state][2][0]:
                        shift = 0
                        if tranOrder[0].input == ' ':
                            conditions[state][2].insert(0, tranOrder[0])
                        elif conditions[state][2][0].input == ' ':
                            shift = 1
                        commonLen = min(len(tranOrder), len(conditions[state][2]))
                        if any(tranOrder[i] != conditions[state][2][i+shift] for i in range(commonLen)):
                            shift = -1 # TODO solve how to merge two sequences
                        if commonLen < len(tranOrder):
                            conditions[state][2].extend(tranOrder[commonLen:-1])
                else:
                    conditions[state][2][:] = tranOrder[:]
        else:
            conditions[state] = (set(), {nextState}, tranOrder.copy(), {lastTran})
        if nextState in conditions:
            conditions[nextState][0].add(state)
        else:
            conditions[nextState] = ({state}, set(), [], set())

    def mapTracesToTransitions(self, traces):
        # mapping of traces to own transitions
        mappedTraces = []
        conditions = {}
        for trace in traces:
            hasNegativeReward = False
            mappedTrace = [0, []]
            state = 0
            lastTran = None
            tranOrder = []
            for fsmTran in trace:
                if fsmTran.reward == REWARD_NEGATIVE:
                    hasNegativeReward = True
                tran = self.getTransition(state, fsmTran.input, fsmTran.output, fsmTran.reward, False)
                nextState = self.getNextState(state, tran)
                if lastTran and lastTran != tran:
                    mappedTrace.append(lastTran)
                    mappedTrace.append(state)
                    mappedTrace.append([])
                    tranOrder.append(lastTran)
                mappedTrace[-1].append(fsmTran)
                lastTran = tran
                if nextState != state:
                    mappedTrace.append(tran)
                    mappedTrace.append(nextState)
                    lastTran = None
                    self.addConditions(conditions, state, tranOrder, tran, nextState)
                    tranOrder = []
                    if nextState != 0:
                        mappedTrace.append([])
                state = nextState
            if len(mappedTrace) > 2:
                mappedTrace.append([]) #not hasNegativeReward) # splitting points
                mappedTrace.append(0) # current position during processing of the mappedTrace
                mappedTraces.append(mappedTrace)
        return mappedTraces, conditions

    def getStateOrder(self, conditions):
        stateOrder = [0]
        statesToAdd = set(conditions[0][1])
        while statesToAdd:
            for ns in statesToAdd:
                if all(state in stateOrder for state in conditions[ns][0]):
                    stateOrder.append(ns)
                    if 0 not in conditions[ns][1]:
                        statesToAdd |= conditions[ns][1]
                    statesToAdd.discard(ns)
                    break
        return stateOrder

    def trySpecializeWithNew(self, fsm, traces):
        mappedTraces, conditions = self.mapTracesToTransitions(traces)
        stateOrder = self.getStateOrder(conditions)
        i = 0
        while i < len(stateOrder):
            nextState = state = stateOrder[i]
            movedState = NULL_STATE
            for k in [2, 3]:    # 2: selfloops state->state, 3: progress transitions
                for relTran in conditions[state][k]:
                    relMappedTraces = [trace for trace in mappedTraces \
                        if trace[trace[-1]] == state and trace[trace[-1]+2] == relTran]
                    transitions = [trace[trace[-1]+1] for trace in mappedTraces \
                        if trace[trace[-1]] == state and trace[trace[-1]+2] == relTran]
                    if movedState != NULL_STATE:
                        nextState = state = movedState
                        tmpRelTran = relTran
                        relTran = self.getTransition(state, relTran.input, relTran.output, relTran.reward, False)
                    if k == 3:
                        nextState = self.getNextState(state, relTran)
                        rewards = set([seq[-1].reward for seq in transitions])
                    elif len(relMappedTraces) > 2 and any(len(trace[trace[-1]+1]) > 1 for trace in relMappedTraces):
                        newState, rewards = self.analyseTransitions(state, relTran, transitions)
                        if newState != state:
                            movedState = newState
                            tmpRelTran = relTran
                            updaterelMappedTraces
                            if rewards:
                                relTran = self.getTransition(state, relTran.input, relTran.output, relTran.reward, False)
                        
                        for spIdx in reversed(relMappedTraces[0][-2]):
                            pos = []
                            neg = []
                            for trace in relMappedTraces:
                                if any([tran.reward == REWARD_NEGATIVE for tran in trace[spIdx+1]]):
                                    neg.append(trace)
                                else:
                                    pos.append(trace)
                            if pos and neg and \
                                ((len(pos) > 2 and any(len(trace[trace[-1]+1]) > 1 for trace in pos)) or \
                                (len(neg) > 2 and any(len(trace[trace[-1]+1]) > 1 for trace in neg))):
                                self.analyseTransitions(s, r, pos)
                                self.analyseTransitions(s, r, neg)
                                if change:
                                    update
                                    break
                            



                    else: # self loops and single transitions
                        rewards = set([seq[-1].reward for seq in transitions])
                    
                    if len(rewards) > 1:
                        if REWARD_NEGATIVE in rewards:


                            stateMap = self.duplicateTransitions(nextState, 0)
                            newTran = relTran.copy()
                            newTran.reward = REWARD_NEGATIVE
                            self.addTransition(state, newTran, stateMap[nextState])
                            if state != nextState:
                                for t, ns in self.transitions[state].items():
                                    if ns == nextState and t != relTran:
                                        newt = t.copy()
                                        newt.reward = REWARD_NEGATIVE
                                        self.addTransition(state, newt, stateMap[nextState])
                                        if checkRewards(t.reward, REWARD_NEGATIVE):
                                            r = REWARD_NEUTRAL if t.reward == REWARD_NONPOSITIVE else REWARD_NONNEGATIVE
                                            self.updateTransition(state, t, r)
                            self.updateMappedTraces(mappedTraces, stateMap)
                            self.updateStateOrder(stateOrder, conditions, stateMap, i)
                            rewards.discard(REWARD_NEGATIVE)
                            if len(rewards) > 1:
                                self.updateTransition(state, relTran, REWARD_NONNEGATIVE)
                            else:
                                self.updateTransition(state, relTran, rewards.pop())
                        else:
                            self.updateTransition(state, relTran, REWARD_NONNEGATIVE)
                    elif rewards:
                        reward = rewards.pop()
                        if reward != REWARD_NEGATIVE and reward != relTran.reward:
                            self.updateTransition(state, relTran, reward)

                    if movedState != NULL_STATE:
                        nextState = state = stateOrder[i]
                        relTran = tmpRelTran
                    for j in range(len(mappedTraces)):
                        if mappedTraces[j][mappedTraces[j][-1]] == state and mappedTraces[j][mappedTraces[j][-1]+2] == relTran:
                            mappedTraces[j][-1] += 3
                            #del mappedTraces[j][:3]
                            #if len(mappedTraces[j]) <= 2:
                                #mappedTraces.pop(j)
            i += 1

    def trySpecializeWith(self, fsm, traces):
        mappedTraces, conditions = self.mapTracesToTransitions(traces)
        stateOrder = self.getStateOrder(conditions)
        i = 0
        while i < len(stateOrder):
            nextState = state = stateOrder[i]
            movedState = NULL_STATE
            for k in [2, 3]:    # 2: selfloops state->state, 3: progress transitions
                for relTran in conditions[state][k]:
                    transitions = [trace[trace[-1]+1] for trace in mappedTraces \
                        if trace[trace[-1]] == state and trace[trace[-1]+2] == relTran]
                    if movedState != NULL_STATE:
                        nextState = state = movedState
                        tmpRelTran = relTran
                        relTran = self.getTransition(state, relTran.input, relTran.output, relTran.reward, False)
                    if k == 3:
                        nextState = self.getNextState(state, relTran)
                        rewards = set([seq[-1].reward for seq in transitions])
                    elif relTran.input == LABEL_OTHERS and len(transitions) > 2 and any(len(seq) > 1 for seq in transitions):
                        newState, rewards = self.analyseTransitions(state, relTran, transitions)
                        if newState != state:
                            movedState = newState
                            tmpRelTran = relTran
                            if rewards:
                                relTran = self.getTransition(state, relTran.input, relTran.output, relTran.reward, False)
                    else: # self loops and single transitions
                        rewards = set([seq[-1].reward for seq in transitions])
                    
                    if len(rewards) > 1:
                        if REWARD_NEGATIVE in rewards:
                            stateMap = self.duplicateTransitions(nextState, 0)
                            newTran = relTran.copy()
                            newTran.reward = REWARD_NEGATIVE
                            self.addTransition(state, newTran, stateMap[nextState])
                            if state != nextState:
                                transitionsToUpdate = []
                                for t, ns in self.transitions[state].items():
                                    if ns == nextState and t != relTran:
                                        transitionsToUpdate.append(t)
                                for t in transitionsToUpdate:
                                    newt = t.copy()
                                    newt.reward = REWARD_NEGATIVE
                                    self.addTransition(state, newt, stateMap[nextState])
                                    if compareRewards(t.reward, REWARD_NEGATIVE):
                                        r = REWARD_NEUTRAL if t.reward == REWARD_NONPOSITIVE else REWARD_NONNEGATIVE
                                        self.updateTransition(state, t, r)
                            self.updateMappedTraces(mappedTraces, stateMap)
                            self.updateStateOrder(stateOrder, conditions, stateMap, i)
                            rewards.discard(REWARD_NEGATIVE)
                            if len(rewards) > 1:
                                self.updateTransition(state, relTran, REWARD_NONNEGATIVE)
                            else:
                                self.updateTransition(state, relTran, rewards.pop())
                        else:
                            self.updateTransition(state, relTran, REWARD_NONNEGATIVE)
                    elif rewards:
                        reward = rewards.pop()
                        if reward != REWARD_NEGATIVE and reward != relTran.reward:
                            self.updateTransition(state, relTran, reward)

                    if movedState != NULL_STATE:
                        nextState = state = stateOrder[i]
                        relTran = tmpRelTran
                    for j in range(len(mappedTraces)):
                        if mappedTraces[j][mappedTraces[j][-1]] == state and mappedTraces[j][mappedTraces[j][-1]+2] == relTran:
                            mappedTraces[j][-1] += 3
                            #del mappedTraces[j][:3]
                            #if len(mappedTraces[j]) <= 2:
                                #mappedTraces.pop(j)
            i += 1
        # merge back where possible

    
    def bla(self):      
        stage = 1
        splitted = False
        while mappedTraces:
            # all segments
            state = mappedTraces[0][0]
            relTran = mappedTraces[0][2]
            nextState = mappedTraces[0][3]
            if stage == 1:
                transitions = [trace[1] for trace in mappedTraces]
                if nextState == 1:
                    stage = 2
                    fsmSeq = max(transitions, key=len)
                    if len(fsmSeq) > 1:
                        fsmTran = fsmSeq[0]
                        tran = self.getTransition(state, fsmTran.input, fsmTran.output, REWARD_ANY, False)
                        self.checkRewards(state, tran, state, [seq[:-1] for seq in transitions if len(seq) > 1])
                    self.checkRewards(state, relTran, nextState, [[seq[-1]] for seq in transitions])
                    # TODO possible split
                else:
                    fsmTran = transitions[0][0]
                    tran = self.getTransition(state, fsmTran.input, fsmTran.output, REWARD_ANY, False)
                    self.checkRewards(state, tran, state, transitions)
                if any(len(seq) > 1 for seq in transitions):
                    self.analyseTransitions(state, relTran, nextState, transitions)
            elif stage == 2:
                # check rewards
                negTransitions = [trace[1] for trace in mappedTraces if trace[3] == nextState]
                if len(negTransitions) != len(mappedTraces):
                    if mappedTraces[0][-1]: #positive
                        posTransitions = negTransitions
                        negTransitions = [trace[1] for trace in mappedTraces if trace[3] != nextState]
                    else:
                        posTransitions = [trace[1] for trace in mappedTraces if trace[3] != nextState]
                    splitted = True
                    if any(seq[-1].reward != REWARD_NEGATIVE for seq in negTransitions):
                        negTransitions #TODO solve inconsistency
                    if any(seq[-1].reward == REWARD_NEGATIVE for seq in posTransitions):
                        posTransitions   #TODO solve inconsistency 
                else:
                    fsmSeq = max(negTransitions, key=len)
                    if len(fsmSeq) > 1:
                        fsmTran = fsmSeq[0]
                        tran = self.getTransition(state, fsmTran.input, fsmTran.output, REWARD_ANY, False)
                        self.checkRewards(state, tran, state, [seq[:-1] for seq in negTransitions if len(seq) > 1])
                    rewards = set([seq[-1].reward for seq in negTransitions])
                    if len(rewards) > 1:
                        if REWARD_NEGATIVE in rewards:
                            #TODO split on the last transition
                            stateMap = self.duplicateTransitions(nextState, 0)
                            newTran = relTran.copy()
                            newTran.reward = REWARD_NEGATIVE
                            self.addTransition(state, newTran, stateMap[nextState])
                            self.updateMappedTraces(mappedTraces, stateMap)
                            rewards.discard(REWARD_NEGATIVE)
                            if len(rewards) > 1:
                                self.updateTransition(state, relTran, REWARD_NONNEGATIVE)
                            else:
                                self.updateTransition(state, relTran, rewards.pop())
                        else:
                            self.updateTransition(state, relTran, REWARD_NONNEGATIVE)
                    else:
                        reward = rewards.pop()
                        if reward != REWARD_NEGATIVE:
                            self.updateTransition(state, relTran, reward)
                stage = 3
            elif stage == 3:
                transitions = [trace[1] for trace in mappedTraces if trace[0] == state]
                if len(transitions) > 2:
                    if nextState == 0:
                        fsmSeq = max(transitions, key=len)
                        if len(fsmSeq) > 1:
                            fsmTran = fsmSeq[0]
                            tran = self.getTransition(state, fsmTran.input, fsmTran.output, REWARD_ANY, False)
                            self.checkRewards(state, tran, state, [seq[:-1] for seq in transitions if len(seq) > 1])
                        self.checkRewards(state, relTran, nextState, [[seq[-1]] for seq in transitions])                    
                    else:
                        fsmTran = transitions[0][0]
                        tran = self.getTransition(state, fsmTran.input, fsmTran.output, REWARD_ANY, False)
                        self.checkRewards(state, tran, state, transitions)
                    if any(len(seq) > 1 for seq in transitions):
                        self.analyseTransitions(state, relTran, nextState, transitions)
                    
                
                ## negative only
                #negSegments = [el for el in allFirstSegments if not el[4]]
                #if len(negSegments) > 1:
                #    self.analyseTransitions(negSegments)
                ## positive only
                #posSegments = [] if len(negSegments) == len(allFirstSegments) else [el for el in allFirstSegments if el[4]]
                #if len(posSegments) > 1:
                #    self.analyseTransitions(posSegments)
                #self.analyseTransitions(allFirstSegments)

            for i in range(len(mappedTraces)-1, -1, -1):
                if mappedTraces[i][0] == state:
                    del mappedTraces[i][:3]
                    if len(mappedTraces[i]) <= 2:
                        mappedTraces.pop(i)


    def trySpecializeWithOld(self, fsm, traces):
        # mapping of traces to own transitions
        mappedTraces = []
        for trace in traces:
            hasNegativeReward = False
            mappedTrace = [0, []]
            state = 0
            for fsmTran in trace:
                if fsmTran.reward == REWARD_NEGATIVE:
                    hasNegativeReward = True
                tran = self.getTransition(state, fsmTran.input, fsmTran.output, fsmTran.reward, False)
                nextState = self.getNextState(state, tran)
                mappedTrace[-1].append(fsmTran)
                if nextState != state:
                    mappedTrace.append(tran)
                    mappedTrace.append(nextState)
                    if nextState != 0:
                        mappedTrace.append([])
                        state = nextState
            if len(mappedTrace) > 2:
                mappedTrace.append(not hasNegativeReward)
                mappedTraces.append(mappedTrace)
        stage = 1
        splitted = False
        while mappedTraces:
            # all segments
            state = mappedTraces[0][0]
            relTran = mappedTraces[0][2]
            nextState = mappedTraces[0][3]
            if stage == 1:
                transitions = [trace[1] for trace in mappedTraces]
                if nextState == 1:
                    stage = 2
                    fsmSeq = max(transitions, key=len)
                    if len(fsmSeq) > 1:
                        fsmTran = fsmSeq[0]
                        tran = self.getTransition(state, fsmTran.input, fsmTran.output, REWARD_ANY, False)
                        self.checkRewards(state, tran, state, [seq[:-1] for seq in transitions if len(seq) > 1])
                    self.checkRewards(state, relTran, nextState, [[seq[-1]] for seq in transitions])
                    # TODO possible split
                else:
                    fsmTran = transitions[0][0]
                    tran = self.getTransition(state, fsmTran.input, fsmTran.output, REWARD_ANY, False)
                    self.checkRewards(state, tran, state, transitions)
                if any(len(seq) > 1 for seq in transitions):
                    self.analyseTransitions(state, relTran, nextState, transitions)
            elif stage == 2:
                # check rewards
                negTransitions = [trace[1] for trace in mappedTraces if trace[3] == nextState]
                if len(negTransitions) != len(mappedTraces):
                    if mappedTraces[0][-1]: #positive
                        posTransitions = negTransitions
                        negTransitions = [trace[1] for trace in mappedTraces if trace[3] != nextState]
                    else:
                        posTransitions = [trace[1] for trace in mappedTraces if trace[3] != nextState]
                    splitted = True
                    if any(seq[-1].reward != REWARD_NEGATIVE for seq in negTransitions):
                        negTransitions #TODO solve inconsistency
                    if any(seq[-1].reward == REWARD_NEGATIVE for seq in posTransitions):
                        posTransitions   #TODO solve inconsistency 
                else:
                    fsmSeq = max(negTransitions, key=len)
                    if len(fsmSeq) > 1:
                        fsmTran = fsmSeq[0]
                        tran = self.getTransition(state, fsmTran.input, fsmTran.output, REWARD_ANY, False)
                        self.checkRewards(state, tran, state, [seq[:-1] for seq in negTransitions if len(seq) > 1])
                    rewards = set([seq[-1].reward for seq in negTransitions])
                    if len(rewards) > 1:
                        if REWARD_NEGATIVE in rewards:
                            #TODO split on the last transition
                            stateMap = self.duplicateTransitions(nextState, 0)
                            newTran = relTran.copy()
                            newTran.reward = REWARD_NEGATIVE
                            self.addTransition(state, newTran, stateMap[nextState])
                            self.updateMappedTraces(mappedTraces, stateMap)
                            rewards.discard(REWARD_NEGATIVE)
                            if len(rewards) > 1:
                                self.updateTransition(state, relTran, REWARD_NONNEGATIVE)
                            else:
                                self.updateTransition(state, relTran, rewards.pop())
                        else:
                            self.updateTransition(state, relTran, REWARD_NONNEGATIVE)
                    else:
                        reward = rewards.pop()
                        if reward != REWARD_NEGATIVE:
                            self.updateTransition(state, relTran, reward)
                stage = 3
            elif stage == 3:
                transitions = [trace[1] for trace in mappedTraces if trace[0] == state]
                if len(transitions) > 2:
                    if nextState == 0:
                        fsmSeq = max(transitions, key=len)
                        if len(fsmSeq) > 1:
                            fsmTran = fsmSeq[0]
                            tran = self.getTransition(state, fsmTran.input, fsmTran.output, REWARD_ANY, False)
                            self.checkRewards(state, tran, state, [seq[:-1] for seq in transitions if len(seq) > 1])
                        self.checkRewards(state, relTran, nextState, [[seq[-1]] for seq in transitions])                    
                    else:
                        fsmTran = transitions[0][0]
                        tran = self.getTransition(state, fsmTran.input, fsmTran.output, REWARD_ANY, False)
                        self.checkRewards(state, tran, state, transitions)
                    if any(len(seq) > 1 for seq in transitions):
                        self.analyseTransitions(state, relTran, nextState, transitions)
                    
                
                ## negative only
                #negSegments = [el for el in allFirstSegments if not el[4]]
                #if len(negSegments) > 1:
                #    self.analyseTransitions(negSegments)
                ## positive only
                #posSegments = [] if len(negSegments) == len(allFirstSegments) else [el for el in allFirstSegments if el[4]]
                #if len(posSegments) > 1:
                #    self.analyseTransitions(posSegments)
                #self.analyseTransitions(allFirstSegments)

            for i in range(len(mappedTraces)-1, -1, -1):
                if mappedTraces[i][0] == state:
                    del mappedTraces[i][:3]
                    if len(mappedTraces[i]) <= 2:
                        mappedTraces.pop(i)


        #stateMap = {0 : 0}
        #statesToCheck = [[0]]
        #while statesToCheck:
        #    fsmStates = statesToCheck.pop(0)
        #    transitions = []
        #    fsmNextStates = []
        #    simStack = []
        #    for fsmState in fsmStates:
        #        simStack.append((fsmState, []))
        #    state = stateMap[fsmStates[0]]
        #    while simStack:
        #        (fsmState, tranSeq) = simStack.pop()
        #        if fsmState in fsm.transitions:
        #            for fsmTran, fsmNS in fsm.transitions[fsmState].items():
        #                tran = self.getTransition(state, fsmTran.input, fsmTran.output, REWARD_ANY, False)
        #                nextState = self.getNextState(state, tran)
        #                if nextState == state:
        #                    simStack.append((fsmNS, tranSeq + [fsmTran]))
        #                else:
        #                    transitions.append(tranSeq)
        #                    fsmNextStates.append(fsmNS)
        #                    stateMap[fsmNS] = nextState
        #    self.analyseTransitions(state, tran, nextState, transitions)
        #    if nextState != 0:
        #        statesToCheck.append(fsmNextStates)
           
          
    def isSpecializationOf(self, other, X=set(), estimated=False):
        if not estimated and self.useOfMapping == MAPPING_FIXED == other.useOfMapping and \
            not set(self.mapping.items()).issubset(set(other.mapping.items())):
            return False
        # simulation
        hasSpecializedTransition = False
        stateMap = {0 : 0}
        statesToCheck = [0]
        while statesToCheck:
            state = statesToCheck.pop(0)
            oState = stateMap[state]
            if state in self.transitions:
                if oState not in other.transitions:
                    return False
                uniqueOutput = None
                uniqueOutputForEach = set()
                outputs = set()
                inputs = Counter()
                for tran, nextState in self.transitions[state].items():
                    t = other.getTransition(oState, tran.input, tran.output, tran.reward, False)
                    if t:
                        if uniqueOutput and uniqueOutput != tran.output:
                            return False
                        if tran.input != t.input or tran.output != t.output or tran.reward != t.reward:
                            if not tran.isSpecializationOf(t):
                                return False
                            hasSpecializedTransition = True
                        if t.input == LABEL_FORALL:
                            if tran.reward == REWARD_NEGATIVE:
                                if t.output == LABEL_EXISTS:
                                    outputs.add(tran.output)
                            elif not uniqueOutput and (t.output == LABEL_EXISTS or len(t.output) == 1) and tran.output != LABEL_EXISTS:
                                uniqueOutput = tran.output 
                        elif t.input == LABEL_EXISTS == t.output:
                            if tran.reward == REWARD_NEGATIVE:
                                inputs[tran.output] += 1
                            else:
                                for o in tran.output:
                                    if o in uniqueOutputForEach:
                                        return False
                                    uniqueOutputForEach.add(o)    
                        ns = other.getNextState(oState, t)
                        if nextState not in stateMap:
                            stateMap[nextState] = ns
                            statesToCheck.append(nextState)
                        elif stateMap[nextState] != ns:
                            return False
                    else:
                        return False # not covered
                if uniqueOutput and self.getTransition(state, None, uniqueOutput, REWARD_NEGATIVE):
                    # there is transition x/uniqueOutput with reward -1 that contradicts other's LABEL_FORALL/LABEL_EXISTS
                    return False
                if X:
                    if (outputs and X == outputs) or (inputs and inputs.most_common(1)[0][1] == len(X)):
                        return False
        return len(self.transitions) == 0 or hasSpecializedTransition or \
            (self.useOfMapping == MAPPING_FIXED and other.useOfMapping != MAPPING_FIXED)
