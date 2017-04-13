from tornado.ioloop import IOLoop
from tornado import httpclient
import datetime

def run():
    def show_response(res):
        print datetime.datetime.now().time(), type(res.body) # res.body[:100]

    requests = []
    for k in xrange(3):
        uri = 'http://localhost:8888/'
        requests.append(uri)

    # # followed by 20 "fast" requests
    # for k in xrange(20):
    #     uri = 'http://localhost:8888/fast?id={}'
    #     requests.append(uri.format(k + 1))

    # show results as they return
    http_client = httpclient.AsyncHTTPClient()

    print 'Scheduling Get Requests:'
    print '------------------------'
    for req in requests:
        print req
        http_client.fetch(req, show_response, request_timeout=600)

    # execute requests on server
    print '\n', datetime.datetime.now().time(), 'Start sending requests....'
    IOLoop.instance().start()

if __name__ == '__main__':
    run()
