from __future__ import unicode_literals

from tornado.web import RequestHandler, Application, StaticFileHandler
from tornado.ioloop import IOLoop
from tornado import gen
from concurrent.futures import ProcessPoolExecutor
import os

import json

import pandas
from csv import QUOTE_ALL

from find_pairs import find_pairs

# http://stackoverflow.com/questions/15375336/how-to-best-perform-multiprocessing-within-requests-with-the-python-tornado-serv

executor = None
df = None
suggested_future = None
rules = []

def find_pairs_wrapper(rules):
    global df
    counts, good_sources, bad_sources = find_pairs(df, rules)

    result = []
    for ((l, r), c) in sorted(counts.iteritems(), key=lambda (k, v): v, reverse=True):
        if abs(c[0]) <= 2 and abs(c[1]) <= 2:
            continue
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
                } for (q1str, q2str, q1t, q2t) in good_sources[(l, r)]
                ],
        }
        result.append(ruleinfo)
    return json.dumps(result)

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

        global suggested_future
        suggested_future = executor.submit(find_pairs_wrapper, rules)
        result = yield suggested_future

        self.set_header('Content-Type', 'application/javascript')
        self.write(result)
        self.flush()

STATIC_PATH = os.path.join(os.getcwd(), "www")

application = Application([
    (r"^/api/suggested", GetSuggestedHandler),
    (r"^/api/addrule", AddRuleHandler),
    (r"^/(.*)$", StaticFileHandler, {"path": STATIC_PATH, "default_filename": "index.html"}),
],
    static_path=STATIC_PATH
)


if __name__ == "__main__":
    application.listen(8888)

    df = pandas.read_csv("train.csv", quoting=QUOTE_ALL)
    df["question2"].fillna("", inplace=True)

    executor = ProcessPoolExecutor(max_workers=3)
    suggested_future = executor.submit(find_pairs_wrapper, rules)

    print "Server reloaded!"

    IOLoop.instance().start()
