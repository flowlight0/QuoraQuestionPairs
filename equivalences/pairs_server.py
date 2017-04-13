from __future__ import unicode_literals

from tornado.web import RequestHandler, Application, StaticFileHandler
from tornado.ioloop import IOLoop
from tornado import gen
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
import os
import traceback, sys

import json
from collections import defaultdict

import pandas
from csv import QUOTE_ALL

from find_pairs import find_pairs, counts_default

# http://stackoverflow.com/questions/15375336/how-to-best-perform-multiprocessing-within-requests-with-the-python-tornado-serv

executor = None
df = None
suggested_future = None
rules = [
    [["how", "can", "i"], ["how", "do", "i"]],
    [["which"], ["what"]],
    [["what's"], ["what", "is"]],
    [["are"], ["is"]],
    [["some"], []],
    [["what", "is", "best", "way", "to"], ["how", "do", "i"]],
    [["really"], []],
    [["so"], []],
    [["good"], ["best"]],
    [["of", "best"], ["best"]],
    [["exactly"], []],
    [["we"], ["i"]],
    [["become"], ["be"]],
    [["how", "does", "one"], ["how", "do", "i"]],
    [["at"], ["in"]],
    [["was"], ["is"]],
    [["of"], ["in"]],
    [["what", "should", "i", "do", "to"], ["how", "do", "i"]],
    [["for"], ["of"]],
    [["ever"], []],
    [["in"], ["on"]],
    [["books"], ["book"]],
    [["what", "does", "it", "feel", "like"], ["what", "is", "it", "like"]],
    [["i"], ["one"]],
    [["differences"], ["difference"]],
    [["great"], ["best"]],
    [["favourite"], ["favorite"]],
    [["top"], ["best"]],
    [["better"], ["best"]],
    [["ways"], ["way"]],
    [["how", "do", "i"], ["how", "to"]],
    [["you've"], ["you", "have"]],
    [["actually"], []],
    [["does"], ["do"]],
    [["is","meant", "by"], ["is"]],
    [["think", "about"], ["think", "of"]],
    [["what", "one", "is"], ["what", "is"]],
    [["what", "can", "i", "do", "to"], ["how", "can", "i"]],
    [["how", "would"], ["how", "do"]],
    [["how", "could"], ["how", "do"]],
    [["companies"], ["company"]],
    [["make", "money"], ["earn", "money"]],
]

rules = [tuple([tuple(x), tuple(y)]) for (x,y) in rules]

NUM_WORKERS = 8

def find_pairs_wrapper(rules, start, stop):
    global df
    try:
        return find_pairs(df.iloc[start:stop, :], rules)
    except:
        print "Exception in find_pairs:"
        print '-'*60
        traceback.print_exc(file=sys.stdout)
        print '-'*60


def find_pairs_fanout():
    global df, rules, suggested_future

    suggested_future = Future()

    futures = [executor.submit(find_pairs_wrapper, rules, i*NUM_WORKERS, (i+1)*NUM_WORKERS) for i in xrange(len(df) / NUM_WORKERS)]

    counts, good_sources, bad_sources = defaultdict(counts_default), defaultdict(set), defaultdict(set)
    for future in as_completed(futures):
        fc, fgs, fbs = future.result()
        for key, v in fc.iteritems():
            counts[key][0] += v[0]
            counts[key][1] += v[1]
        for key, v in fgs.iteritems():
            if len(good_sources[key]) < 5:
                good_sources[key] = set(list(good_sources[key] | v)[:5])
        for key, v in fbs.iteritems():
            if len(bad_sources[key]) < 5:
                bad_sources[key] = set(list(bad_sources[key] | v)[:5])

    result = []
    for ((l, r), c) in sorted(counts.iteritems(), key=lambda (k, v): v, reverse=True):
        if abs(c[0]) <= 2 and abs(c[1]) <= 2:
            continue
        if len(result) > 100:
            break
        ruleinfo = {
            "counts": c,
            "left": l,
            "right": r,
            "good_examples": [
                {
                    "q1_orig": q1str,
                    "q2_orig": q2str,
                    "q1_simp": " ".join(q1t),
                    "q2_simp": " ".join(q2t)
                } for (q1str, q2str, q1t, q2t) in good_sources[(l, r)]
                ],
            "bad_examples": [
                {
                    "q1_orig": q1str,
                    "q2_orig": q2str,
                    "q1_simp": " ".join(q1t),
                    "q2_simp": " ".join(q2t)
                } for (q1str, q2str, q1t, q2t) in bad_sources[(l, r)]
                ],
        }
        result.append(ruleinfo)

    suggested_future.set_result(json.dumps(result))

class GetSuggestedHandler(RequestHandler):
    @gen.coroutine
    def get(self):
        global suggested_future
        result = yield suggested_future
        self.set_header('Content-Type', 'application/javascript')
        self.write(result)
        self.flush()

class AddRuleHandler(RequestHandler):
    @gen.coroutine
    def post(self):
        global rules
        rule = json.loads(self.request.body)
        rules.append((tuple(rule["left"]), tuple(rule["right"])))

        find_pairs_fanout()
        global suggested_future
        result = yield suggested_future

        self.set_header('Content-Type', 'application/javascript')
        self.write(result)
        self.flush()

class GetRulesHandler(RequestHandler):
    def get(self):
        global rules
        self.set_header('Content-Type', 'application/javascript')
        self.write(json.dumps(rules))
        self.flush()

STATIC_PATH = os.path.join(os.getcwd(), "www")

application = Application([
    (r"^/api/suggested", GetSuggestedHandler),
    (r"^/api/addrule", AddRuleHandler),
    (r"^/api/rules", GetRulesHandler),
    (r"^/(.*)$", StaticFileHandler, {"path": STATIC_PATH, "default_filename": "index.html"}),
],
    static_path=STATIC_PATH
)


if __name__ == "__main__":
    application.listen(8888)

    df = pandas.read_csv("train.csv", quoting=QUOTE_ALL)
    df["question2"].fillna("", inplace=True)

    executor = ProcessPoolExecutor(max_workers=NUM_WORKERS)
    find_pairs_fanout()

    print "Server reloaded!"

    IOLoop.instance().start()
