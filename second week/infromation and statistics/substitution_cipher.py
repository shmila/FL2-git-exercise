import matplotlib.pyplot as plt
import urllib2
import random
import numpy as np


def get_n_grams(s, n):
    for i in range(len(s) - n):
        yield s[i:i + n]


def get_n_choise(l, n):
    for e in l:
        if n == 1:
            yield e
        else:
            for s0 in get_n_choise(l, n - 1):
                yield e + s0


def get_n_hist(text, n):
    alephbeth = sorted(list(set(text.lower())))
    text = text.lower()
    alephbeth_n = list(get_n_choise(alephbeth, n))
    if n > 2:
        alephbeth_2 = list(get_n_choise(alephbeth, 2))
        d_count_2 = {c: text.count(c) for c in alephbeth_2}
        d_count = {k: 0 for k in alephbeth_n}
        d_count.update({c: text.count(c) for c in alephbeth_n
                        if d_count_2.get(c[:2], 0) > 0 and d_count_2.get(c[-2:], 0) > 0})

    else:
        d_count = {c: text.count(c) for c in alephbeth_n}
    N = sum(d_count.values())
    return {k: 1.0 * d_count[k] / N for k in d_count}


random.seed(0)

path = 'http://www.gutenberg.org/files/1524/1524-0.txt'
path_corpus = 'http://www.gutenberg.org/cache/epub/1120/pg1120.txt'

text = urllib2.urlopen(path).read()
corpus = urllib2.urlopen(path_corpus).read()

alephbeth = [chr(i) for i in range(ord('a'), ord('z') + 1)]

text0 = ''.join(map((lambda c: c if c in alephbeth else ''), text.lower()))
corpus0 = ''.join(map((lambda c: c if c in alephbeth else ''), corpus.lower()))

encrpted_alephbeth = map(str.upper, alephbeth)
random.shuffle(encrpted_alephbeth)

text1 = ''.join(encrpted_alephbeth[alephbeth.index(i)] for i in text0)

encrypt_c = lambda x: encrpted_alephbeth[alephbeth.index(x.lower())]
decrypt_c = lambda x: alephbeth[encrpted_alephbeth.index(x.upper())]

decrypt = lambda x: ''.join(map(decrypt_c, x))
encrypt = lambda x: ''.join(map(encrypt_c, x))

text1_hist = get_n_hist(text1[20000:25000], 2)
print decrypt(max(text1_hist.items(), key=lambda x: x[1])[0])
