from http import server
from threading import Condition
import base64
import io
import logging
import os
import socketserver

import numpy as np
import picamera
import argparse
import time
import signal

import edgetpu.classification.engine
# from edgetpu.detection.engine import DetectionEngine


# Parameters
AUTH_USERNAME = os.environ.get('AUTH_USERNAME', 'balena')
AUTH_PASSWORD = os.environ.get('AUTH_PASSWORD', 'balena')
AUTH_BASE64 = base64.b64encode('{}:{}'.format(AUTH_USERNAME, AUTH_PASSWORD).encode('utf-8'))
BASIC_AUTH = 'Basic {}'.format(AUTH_BASE64.decode('utf-8'))
RESOLUTION = os.environ.get('RESOLUTION', '640x480').split('x')
RESOLUTION_X = int(RESOLUTION[0])
RESOLUTION_Y = int(RESOLUTION[1])
ROTATION = int(os.environ.get('ROTATE', 0))
HFLIP = os.environ.get('HFLIP', 'false').lower() == 'true'
VFLIP = os.environ.get('VFLIP', 'false').lower() == 'true'

PAGE = """\
<html>
<head>
<title>edgeTPU object identification</title>
</head>
<body>
<h1>edgeTPU object identification</h1>
<img src="stream.mjpg" width="{}" height="{}" />
</body>
</html>
""".format(RESOLUTION_X, RESOLUTION_Y)


class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()
        self.engine = None
        self.obj_engine = None

    def set_engine(self, engine):
        self.engine = engine
    
    def set_obj_engine(self, obj_engine):
        self.obj_engine = obj_engine

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Authorization') is None:
            self.do_AUTHHEAD()
            self.wfile.write(b'no auth header received')
        elif self.headers.get('Authorization') == BASIC_AUTH:
            self.authorized_get()
        else:
            self.do_AUTHHEAD()
            self.wfile.write(b'not authenticated')

    def do_AUTHHEAD(self):
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm=\"picamera\"')
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def authorized_get(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                stream_video = io.BytesIO()
                stream_tpu = io.BytesIO()
                _, width, height, channels = engine.get_input_tensor_shape()
                print("tensor width = ", width)
                print("tensor height = ", height)
                while True:
                    camera.capture(stream_tpu,
                                        format='rgb',
                                        use_video_port=True,
                                        resize=(width, height))

                    stream_tpu.truncate()
                    stream_tpu.seek(0)
                    input = np.frombuffer(stream_tpu.getvalue(), dtype=np.uint8)
                    start_ms = time.time()
                    results = engine.ClassifyWithInputTensor(input, top_k=1, threshold=0.5)
                    # objects = obj_engine.DetectWithInputTensor(input, top_k=1, threshold=0.5)
                    
                    if results:
                        camera.annotate_text = "%s %.2f" % (labels[results[0][0]], results[0][1])
                    else:
                        camera.annotate_text = ""
                    
                    # if objects:
                    #     print("%s %.2f" % (obj_labels[objects[0].label_id], objects[0].score))
                        
                    camera.capture(stream_video, format='jpeg', use_video_port=True)
                    stream_video.truncate()
                    stream_video.seek(0)

                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(stream_video.getvalue()))
                    self.end_headers()
                    self.wfile.write(stream_video.getvalue())
                    self.wfile.write(b'\r\n')

            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def sigterm_handler(signal, frame):
    camera.close()
    server.shutdown()
    print('Exiting...')
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGINT, sigterm_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='File path of Tflite model.', required=True)
    parser.add_argument('--label', help='File path of label file.', required=True)
    # parser.add_argument("--objmodel", default="models/mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite", help="Path of the detection model.")
    # parser.add_argument("--objlabel", default="models/coco_labels.txt", help="Path of the labels file.")

    args = parser.parse_args()
    res = '{}x{}'.format(RESOLUTION_X, RESOLUTION_Y)

    with open(args.label, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)
    
    # with open(args.objlabel, 'r') as f:
    #     pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
    #     obj_labels = dict((int(k), v) for k, v in pairs)

    engine = edgetpu.classification.engine.ClassificationEngine(args.model)
    # obj_engine = DetectionEngine(args.objmodel)

    with picamera.PiCamera(resolution='640x480', framerate=24) as camera:
        camera.hflip = HFLIP
        camera.vflip = VFLIP
        camera.rotation = ROTATION

        try:
            address = ('', 80)
            server = StreamingServer(address, StreamingHandler)
            server.serve_forever()
        finally:
            camera.close()
            server.shutdown()