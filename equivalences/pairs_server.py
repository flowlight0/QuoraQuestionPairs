from __future__ import unicode_literals

from tornado.web import RequestHandler, Application
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


def find_pairs_wrapper():
    global df
    return find_pairs(df)

class IndexHandler(RequestHandler):
    @gen.coroutine
    def get(self):
        global executor, df
        self.set_header('Content-Type', 'application/javascript')
        counts, good_sources, bad_sources = yield executor.submit(find_pairs_wrapper)

        result = []
        for ((l, r), c) in sorted(counts.iteritems(), key=lambda (k, v): v, reverse=True):
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
        self.write(json.dumps(result))
        self.flush()


application = Application([
    (r"/", IndexHandler),
], static_path=os.path.join(os.getcwd(), "static")
)


if __name__ == "__main__":
    application.listen(8888)

    df = pandas.read_csv("train.csv", quoting=QUOTE_ALL)
    df["question2"].fillna("", inplace=True)

    executor = ProcessPoolExecutor(max_workers=3)

    print "Server reloaded!"

    IOLoop.instance().start()
