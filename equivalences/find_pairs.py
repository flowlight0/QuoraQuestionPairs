from __future__ import unicode_literals

import pandas
import re
import itertools
from csv import QUOTE_ALL
from collections import defaultdict
from stopwords import process, build_anchors
from Levenshtein import distance


def fix(stopwords, q):
    return [x for x in re.findall(r"[\w\d][\w\d'-]*", q.decode("utf-8").lower(), re.UNICODE) if x not in stopwords]


def best_match(stopwords, anchors, q1, q2):
    q1set = process(stopwords, anchors, tuple(q1))
    q2set = process(stopwords, anchors, tuple(q2))
    cross = []

    for q1tok in q1set:
        q1str = " ".join(q1tok)
        for q2tok in q2set:
            q2str = " ".join(q2tok)
            if q1str.find(q2str) != -1 or q2str.find(q1str) != -1 or q1tok == q2tok:
                return None
            cross.append((q1str, q2str, q1tok, q2tok))

    return sorted(cross, key=lambda (s1, s2, t1, t2): distance(s1, s2))[0][2:]


def find_pairs(df):
    stopwords = set(['a', 'the', 'an'])

    rules = [
        (("what's",), ("what", "is")),
        (("what", "are"), ("what", "is")),
        (("what", "are", "some"), ("what", "is")),

        (("which", "are"), ("which", "is")),
        (("which", "are", "some"), ("which", "is")),

        (("which", "is", "best"), ("what", "is")),

        (("how", "can", "i"), ("how", "to")),
        (("how", "do", "i"), ("how", "to")),
        (("what", "is", "best", "way", "to"), ("how", "to")),

        (("my",), ()),
    ]
    anchors = build_anchors(rules)

    # rule => set of (q1 orig text, q2 orig text, q1 simplified, q2 simplified)
    good_sources = defaultdict(set)
    # rule => set of (q1 orig text, q2 orig text, q1 simplified, q2 simplified)
    bad_sources = defaultdict(set)
    # rule => [good counts, bad counts]
    counts = {}

    for line in df.itertuples():
        q1 = fix(stopwords, line.question1)
        q2 = fix(stopwords, line.question2)

        matches = set()

        pair = best_match(stopwords, anchors, q1, q2)
        if pair is None:
            continue
        q1, q2 = pair

        left = len(list(itertools.takewhile(
            lambda (x, y): x == y,
            zip(q1, q2)
        )))
        right = len(list(itertools.takewhile(
            lambda (x, y): x == y,
            zip(reversed(q1), reversed(q2))
        )))

        if right > 0:
            left_match, right_match = sorted([tuple(q1[left:-right]), tuple(q2[left:-right])])
        else:
            left_match, right_match = sorted([tuple(q1[left:]), tuple(q2[left:])])
        if left_match or right_match:
            matches.add((left_match, right_match))

        for key in matches:
            if int(line.is_duplicate):
                sources = good_sources
                idx = 0
                c = 1
            else:
                sources = bad_sources
                idx = 1
                c = -1

            if len(sources[key]) < 5:
                sources[key].add((line.question1, line.question2, q1, q2))
            if key not in counts:
                counts[key] = [0, 0]
            counts[key][idx] += c
    return counts, good_sources, bad_sources


def main():
    df = pandas.read_csv("train.csv", quoting=QUOTE_ALL)
    df["question2"].fillna("", inplace=True)

    counts, good_sources, bad_sources = find_pairs(df)

    for ((l, r), c) in sorted(counts.iteritems(), key=lambda (k, v): v, reverse=True):
        print c, "\t", l, "=>", r
        print "GOOD:"
        for (q1str, q2str, q1t, q2t) in good_sources[(l, r)]:
            print q1str, "\t\t", q2str
            print "\t", " ".join(q1t), "\t\t", " ".join(q2t)
        print "BAD:"
        for (q1str, q2str, q1t, q2t) in bad_sources[(l, r)]:
            print q1str, "\t\t", q2str
        print


if __name__ == "__main__":
    main()
