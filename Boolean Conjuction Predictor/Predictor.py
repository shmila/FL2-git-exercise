import sys


def evaluate(h, t):
    res = 1
    for i in range(len(h)):
        if (h[i] == 1):
            if (i % 2 == 0):
                # temp = int(t[i / 2])
                res = res * (t[i / 2])
            else:
                # temp = int(1 - t[i / 2])
                res = res * (1 - t[i / 2])
    return res


def all_negative_hypothesis(n):
    return [1 for i in range(2 * n)]


def pretty_print(h):
    res = ""
    for i in range(len(h)):
        if int(h[i]) == 1:
            if i % 2 == 0:
                res += ("x" + str(i / 2 + 1))
            else:
                res += ("not(x" + str((i + 1) / 2) + ")")
            if i < (len(h) - 2):
                res += ","
    with open("shmila_output.txt", 'w+') as out:
        out.write(res)


def consistency_algorithm(examples):
    n = len(examples[0]) - 1
    h = all_negative_hypothesis(n)
    for t in examples:
        predicted_value = evaluate(h, t)
        actual_value = t[-1]
        if actual_value == 1 and predicted_value == 0:
            for i in range(n):
                if t[i] == 1:
                    h[2 * i + 1] = 0
                else:
                    h[2 * i] = 0
        if actual_value == 0:
            continue
    return h


def main():
    if len(sys.argv) < 2:
        print("too few arguments! expected input data file as an argument.")
    else:
        examples = []
        with open(sys.argv[1], 'r') as data:
            lines = data.readlines()
            for line in lines:
                line = line.rstrip()
                examples.append(map(int, line.split(' ')))
            h = consistency_algorithm(examples)
            pretty_print(h)


if __name__ == "__main__":
    main()
