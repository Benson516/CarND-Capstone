#!/usr/bin/env python

from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

import socketio
import time

from bridge import Bridge
from conf import conf

#
import threading

sio = socketio.Server(async_mode='gevent')
msgs = []

dbw_enable = False

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

def send(topic, data):
    sio.emit(topic, data=data, skip_sid=True)

bridge = Bridge(conf, send)

@sio.on('telemetry')
def telemetry(sid, data):
    global dbw_enable
    if data["dbw_enable"] != dbw_enable:
        dbw_enable = data["dbw_enable"]
        bridge.publish_dbw_status(dbw_enable)
    # bridge.publish_odometry(data)
    _t = threading.Thread(target=bridge.publish_odometry, args=(data,) )
    # _t.daemon = True
    _t.start()

@sio.on('control')
def control(sid, data):
    bridge.publish_controls(data)

@sio.on('obstacle')
def obstacle(sid, data):
    bridge.publish_obstacles(data)

@sio.on('lidar')
def obstacle(sid, data):
    bridge.publish_lidar(data)

@sio.on('trafficlights')
def trafficlights(sid, data):
    bridge.publish_traffic(data)

@sio.on('image')
def image(sid, data):
    # _t_s = time.time()
    bridge.publish_camera(data)
    # _d_1 = time.time() - _t_s
    # print("_d_1 = %f" % _d_1)
    #
    # _t = threading.Thread(target=bridge.publish_camera, args=(data,) )
    # # _t.daemon = True
    # _t.start()

if __name__ == '__main__':

    # Create socketio WSGI application
    app = socketio.WSGIApp(sio)

    # deploy as an gevent WSGI server
    pywsgi.WSGIServer(('', 4567), app, handler_class=WebSocketHandler).serve_forever()
