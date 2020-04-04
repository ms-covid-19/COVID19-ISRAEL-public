import os

import sys
# sys.path = [i for i in sys.path if '/home' not in i]
sys.path.append(os.path.abspath(os.path.join(__file__, 'assets')))
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

import dash
import logging

from Dashboard.app import app
server = app.server
log_ = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


import Dashboard.callbacks

if __name__ == '__main__':
    log_.info('Starting the app!')
    app.server.run(port=4443, debug=False)
