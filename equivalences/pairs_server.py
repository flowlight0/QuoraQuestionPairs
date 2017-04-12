from tornado.web import RequestHandler, Application
from tornado.ioloop import IOLoop
from tornado import gen
from concurrent.futures import ProcessPoolExecutor

import json

import pandas
from csv import QUOTE_ALL

from find_pairs import find_pairs

# http://stackoverflow.com/questions/15375336/how-to-best-perform-multiprocessing-within-requests-with-the-python-tornado-serv

executor = None
df = None


class IndexHandler(RequestHandler):
    @gen.coroutine
    def get(self):
        global executor, df
        self.set_header('Content-Type', 'application/javascript')
        hw_results = yield executor.submit(find_pairs, df)
        self.write(json.dumps(hw_results))
        self.flush()


application = Application([
    (r"/", IndexHandler),
])


if __name__ == "__main__":
    application.listen(8888)

    executor = ProcessPoolExecutor(max_workers=3)

    df = pandas.read_csv("train.csv", quoting=QUOTE_ALL)
    df["question2"].fillna("", inplace=True)

    print "Server reloaded!"

    IOLoop.instance().start()
