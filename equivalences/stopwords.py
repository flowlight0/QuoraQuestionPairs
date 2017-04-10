import sys
import pandas
import re
from csv import QUOTE_ALL
from Levenshtein import distance
import heapq

MAX_GENERATION = 5

def fix(q):
    return tuple(re.findall(r"[\w\d][\w\d'-]*", q.lower(), re.UNICODE))

def process(stopwords, anchors, q):
    assert isinstance(q, tuple)
    return process_impl(stopwords, anchors, [(0, q)])

def process_impl(stopwords, anchors, queue, seen=None):
    if not queue:
        return seen
    if seen is None:
        seen = set()
    while True:
        gen, q = heapq.heappop(queue)
        q = tuple(w for w in q if w not in stopwords)
        if gen >= MAX_GENERATION:
            seen.add(q)
        if q in seen:
            if not queue:
                return seen
            continue
        break
    seen.add(q)

    good_rules = []
    for i, w in enumerate(q):
        if w not in anchors:
            continue
        for rule in anchors[w]:
            apply = True
            for j in xrange(len(rule[0])):
                if i + j >= len(q) or rule[0][j] != q[i + j]:
                    apply = False
                    break
            if apply:
                good_rules.append((i, rule))

    for gr in good_rules:
        q1 = list(q)
        q1[gr[0]:gr[0] + len(gr[1][0])] = list(gr[1][1])
        q1 = tuple(q1)
        if q1 not in seen:
            heapq.heappush(queue, (gen + 1, q1))
    return process_impl(stopwords, anchors, queue, seen)

def build_anchors(rules):
    anchors = dict()
    for r in rules:
        if r[0][0] not in anchors:
            anchors[r[0][0]] = list()
        anchors[r[0][0]].append(r)
    return anchors

def main():
    df = pandas.read_csv(sys.argv[1], quoting=QUOTE_ALL)
    df["question2"].fillna("", inplace=True)
    stopwords = set(['a', 'the'])
    rules = []
    anchors = build_anchors(rules)

    for line in df.itertuples():
        if not line.is_duplicate:
            continue
        q1 = fix(line.question1)
        LEN1 = len(" ".join(q1)) + 5
        q2 = fix(line.question2)
        LEN2 = len(" ".join(q2)) + 5
        fmtstring = "%" + str(LEN1) + "s      %" + str(LEN2) + "s"
        print
        print
        print (">" + fmtstring) % (" ".join(q1), " ".join(q2))
        while True:
            q1set = process(stopwords, anchors, q1)
            q2set = process(stopwords, anchors, q2)
            q1best = None
            q2best = None
            cross = []

            match = False
            for q1str in q1set:
                q1str = " ".join(q1str)
                for q2str in q2set:
                    q2str = " ".join(q2str)
                    if q1str.find(q2str) != -1 or q2str.find(q1str) != -1 or set(q1str.split()) == set(q2str.split()):
                        match = True
                        q1best = q1str
                        q2best = q2str
                        break
                    cross.append((q1str, q2str))
                if match:
                    break

            if match:
                print ("!" + fmtstring + "      MATCH!") % (q1best, q2best)
                break
            else:
                cross = sorted([(distance(s1, s2), s1, s2) for s1, s2 in cross], reverse=True)
                for mindist, s1, s2 in cross[-7:]:
                    print ("*" + fmtstring + "      %d") % (s1, s2, mindist)

            cmd = raw_input()
            if cmd == "":
                break
            elif cmd[0] == "+":
                stopwords.add(cmd[1:])
            elif cmd[0] == "-":
                stopwords.discard(cmd[1:])
            elif cmd == "m":
                source = raw_input().split()
                target = raw_input().split()
                if not source or not target:
                    continue
                rule = (source, target)
                rules.append(rule)
                if source[0] not in anchors:
                    anchors[source[0]] = list()
                anchors[source[0]].append(rule)
            elif cmd == "p":
                print
                print stopwords
                print
                print rules
                print

if __name__ == "__main__":
    main()
