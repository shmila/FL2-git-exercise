import numpy as np


def boolean_conj_predictor(input_file):
    return None


def loadFile(input_file):
    data = np.loadtxt(input_file)
    x = data[:, :-1].T
    y = data[:, -1]

    # trainingSet = []
    # for row in data:
    #     inst = row[0:-1]
    #     tag = row[-1]
    #     trainingSet.append((inst,tag))
    # # OLD LOAD FILE
    # f = open(input_file)
    # trainingSet = []
    # for line in f:
    #     inst = [int(char) for char in line[0:-2:2]]
    #     print("inst:  " + str(inst))
    #     # print("type inst:  " + str(type(inst)))
    #     # print(l_txt)
    #     # print(l_txt[-1])
    #     tag = line[-2]
    #     print(tag)
    #     trainingSet.append((inst, tag))
    return (x,y)

def eval(inst, hypo):
    evaluated_tag = 1
    idx = 0
    while evaluated_tag and idx < len(inst):  # hypo is not rejected and we didn't fix all operands
        if len(hypo[idx]) == 1:  # if it has unique operator on this literal -> evaluate
            evaluated_tag = evaluated_tag * (inst[idx] == hypo[idx][0])
        elif len(hypo[idx]) > 1:  # if there is more than a unique operator on this literal the hypo can NOT be right
            evaluated_tag = 0
        idx += 1
    return evaluated_tag

def learn(x, y):
    dim = x.shape[0]
    instNum = x.shape[1]
    hypo = [[1,0] for litrl in range(dim)]  # each literal as a set of both positive and negative operandsso it is mutable
    for idx in range(instNum):  # go over all instances
        inst = x[:, idx]  # take instance as a row
        evalTag = eval(inst, hypo)  # evaluate instance with current hypothesis
        if y[idx] == 1 and evalTag == 0:  # if tag is 1 and evaluation is zero -> fix
            for litrl_idx in range(dim):  # go over literals by index
                if inst[litrl_idx] == 1 and 0 in hypo[litrl_idx]:  # if (the literal is one) AND (negative operand is on in hypo list)
                    hypo[litrl_idx].remove(0)  # turn off operand!
                elif inst[litrl_idx] == 0 and 1 in hypo[litrl_idx]:  # if (the literal is zero) AND (positive operand is on in hypo list)
                    hypo[litrl_idx].remove(1)  # turn off operand!
    return hypo

def returnHypo(hypo):
    resultStr = ''
    for litr in enumerate(hypo):  # go over literal by index
        litrStr = ''
        if litr[1] == [0]:  # negative operand is on
            litrStr = litrStr + 'not(x' + str(litr[0]) + '),'  # write operand and index
        elif litr[1] == [1]:  # positive operand is on
            litrStr = litrStr + 'x' + str(litr[0]) + ','  # write operand and index
        resultStr = resultStr + litrStr
    result = resultStr[:-1]  # remove last comma
    return result


if __name__ == "__main__":
    x , y = loadFile("data.txt")
    hypo = learn(x,y)
    hypoStr = returnHypo(hypo)
    print("Hypothesis is: " + hypoStr)
