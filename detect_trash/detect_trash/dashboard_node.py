#!/usr/bin/env python3

import json
import math
import threading
import time

import cv2
import numpy as np
import rclpy
from flask import Flask, Response, jsonify
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String


HTML_PAGE = r'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Waste Sorting Manipulator Dashboard</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Arial,"Noto Sans KR",sans-serif;background:#0f172a;color:#e2e8f0}header{height:72px;display:flex;align-items:center;justify-content:space-between;padding:0 28px;background:#111827;border-bottom:1px solid #263247}header h1{margin:0;font-size:22px}.sub{margin-top:4px;color:#94a3b8;font-size:13px}.ros{display:flex;align-items:center;gap:9px;font-size:14px;font-weight:700}.dot{width:10px;height:10px;border-radius:50%;background:#ef4444}.dot.on{background:#22c55e;box-shadow:0 0 12px rgba(34,197,94,.65)}main{padding:22px;max-width:1500px;margin:auto}.grid{display:grid;grid-template-columns:minmax(560px,1.6fr) minmax(360px,1fr);gap:18px}.card{background:#111827;border:1px solid #263247;border-radius:14px;padding:18px;box-shadow:0 8px 30px rgba(0,0,0,.18)}.card h2{font-size:16px;margin:0 0 14px;color:#f8fafc}.video{background:#020617;border-radius:10px;overflow:hidden;aspect-ratio:4/3;display:flex;align-items:center;justify-content:center}.video img{width:100%;height:100%;object-fit:contain}.counts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}.count{padding:14px 12px;background:#0b1220;border:1px solid #263247;border-radius:11px}.count .label,.metric .label{font-size:12px;color:#94a3b8;margin-bottom:8px}.count .value{font-size:28px;font-weight:800}.paper{color:#f87171}.pet{color:#60a5fa}.can{color:#4ade80}.target{border-left:4px solid #f59e0b;background:#0b1220;border-radius:10px;padding:15px}.target-name{font-size:24px;font-weight:800;margin-bottom:14px;text-transform:uppercase}.target-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.metric{background:#111827;border:1px solid #263247;border-radius:9px;padding:11px}.metric .value{margin-top:5px;font-size:17px;font-weight:700}.bottom{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid #263247}th{color:#94a3b8}.pill{display:inline-block;padding:4px 9px;border-radius:999px;font-size:11px;font-weight:800}.onp{background:rgba(34,197,94,.14);color:#4ade80}.offp{background:rgba(239,68,68,.14);color:#f87171}.joints{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.joint{background:#0b1220;border:1px solid #263247;border-radius:9px;padding:12px}.joint .name{color:#94a3b8;font-size:12px}.joint .angle{margin-top:6px;font-size:19px;font-weight:800}.muted{color:#64748b}.foot{margin-top:14px;color:#64748b;font-size:11px}@media(max-width:1000px){.grid,.bottom{grid-template-columns:1fr}}
</style>
</head>
<body>
<header><div><h1>Waste Sorting Manipulator Dashboard</h1><div class="sub">YOLO · ArUco Homography · ROS 2 · OpenManipulator-X</div></div><div class="ros"><span id="dot" class="dot"></span><span id="system">WAITING</span></div></header>
<main>
<section class="grid">
<div class="card"><h2>Live Detection</h2><div class="video"><img src="/video_feed" alt="ROS Camera Stream"></div></div>
<div class="card"><h2>Detected Objects</h2><div class="counts"><div class="count"><div class="label">PAPER</div><div id="paper" class="value paper">0</div></div><div class="count"><div class="label">PET</div><div id="pet" class="value pet">0</div></div><div class="count"><div class="label">CAN</div><div id="can" class="value can">0</div></div></div><h2>Nearest Target</h2><div class="target"><div id="tname" class="target-name">WAITING</div><div class="target-grid"><div class="metric"><div class="label">Class</div><div id="tclass" class="value">-</div></div><div class="metric"><div class="label">Confidence</div><div id="tconf" class="value">-</div></div><div class="metric"><div class="label">Robot X</div><div id="tx" class="value">-</div></div><div class="metric"><div class="label">Robot Y</div><div id="ty" class="value">-</div></div><div class="metric"><div class="label">Distance</div><div id="td" class="value">-</div></div><div class="metric"><div class="label">Pixel</div><div id="tp" class="value">-</div></div></div></div></div>
</section>
<section class="bottom"><div class="card"><h2>ROS Topic Status</h2><table><thead><tr><th>Topic</th><th>Status</th><th>Age</th></tr></thead><tbody id="topics"></tbody></table></div><div class="card"><h2>Robot Joint State</h2><div id="joints" class="joints"><div class="muted">/joint_states 대기 중...</div></div></div></section>
<section class="card" style="margin-top:18px"><h2>Detection List</h2><table><thead><tr><th>Type</th><th>Class</th><th>Confidence</th><th>Pixel X</th><th>Pixel Y</th></tr></thead><tbody id="dets"><tr><td colspan="5" class="muted">객체 인식 결과 대기 중...</td></tr></tbody></table><div class="foot">ROS 2 Topics → Flask HTTP API / MJPEG Stream</div></section>
</main>
<script>
function age(v){return v==null?'-':v.toFixed(2)+' s'}
function row(name,x){let c=x.online?'onp':'offp',t=x.online?'ONLINE':'OFFLINE';return `<tr><td>${name}</td><td><span class="pill ${c}">${t}</span></td><td>${age(x.age)}</td></tr>`}
async function refresh(){try{let r=await fetch('/api/status',{cache:'no-store'}),d=await r.json();paper.textContent=d.counts.paper||0;pet.textContent=d.counts.pet||0;can.textContent=d.counts.can||0;let t=d.nearest_target;if(t){tname.textContent=t.type||'UNKNOWN';tclass.textContent=t.class||'-';tconf.textContent=t.confidence!==undefined?(t.confidence*100).toFixed(1)+'%':'-';tx.textContent=t.x!==undefined?(t.x*100).toFixed(1)+' cm':'-';ty.textContent=t.y!==undefined?(t.y*100).toFixed(1)+' cm':'-';td.textContent=t.distance!==undefined?(t.distance*100).toFixed(1)+' cm':'-';tp.textContent=(t.pixel_x!==undefined&&t.pixel_y!==undefined)?`(${t.pixel_x}, ${t.pixel_y})`:'-'}else{tname.textContent='WAITING';tclass.textContent=tconf.textContent=tx.textContent=ty.textContent=td.textContent=tp.textContent='-'}topics.innerHTML=row('/detect_trash/image_raw',d.topics.image_raw)+row('/detect_trash/detections',d.topics.detections)+row('/detect_trash/nearest_target',d.topics.nearest_target)+row('/joint_states',d.topics.joint_states);let ok=d.topics.image_raw.online&&d.topics.detections.online;dot.classList.toggle('on',ok);system.textContent=ok?'ROS DATA ONLINE':'WAITING FOR ROS DATA';joints.innerHTML=d.joints.length?d.joints.map(j=>`<div class="joint"><div class="name">${j.name}</div><div class="angle">${j.position_deg.toFixed(1)}°</div></div>`).join(''):'<div class="muted">/joint_states 대기 중...</div>';dets.innerHTML=d.detections.length?d.detections.map(x=>`<tr><td>${x.type||'-'}</td><td>${x.class||'-'}</td><td>${((x.confidence||0)*100).toFixed(1)}%</td><td>${x.center_x??'-'}</td><td>${x.center_y??'-'}</td></tr>`).join(''):'<tr><td colspan="5" class="muted">현재 검출된 객체 없음</td></tr>'}catch(e){dot.classList.remove('on');system.textContent='DASHBOARD ERROR'}}
setInterval(refresh,500);refresh();
</script>
</body>
</html>'''


class WasteDashboardNode(Node):

    def __init__(self):
        super().__init__('waste_dashboard')
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.detections = []
        self.nearest_target = None
        self.joint_names = []
        self.joint_positions = []
        self.last_image_time = None
        self.last_detection_time = None
        self.last_target_time = None
        self.last_joint_time = None

        self.create_subscription(Image, '/detect_trash/image_raw', self.image_callback, qos_profile_sensor_data)
        self.create_subscription(String, '/detect_trash/detections', self.detection_callback, 10)
        self.create_subscription(String, '/detect_trash/nearest_target', self.target_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_callback, qos_profile_sensor_data)

        self.app = Flask(__name__)
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/api/status', 'api_status', self.api_status)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)

        self.get_logger().info('Waste Dashboard 시작')
        self.get_logger().info('Browser : http://127.0.0.1:5000')

    def image_callback(self, msg):
        if msg.encoding != 'bgr8':
            return
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8)
            frame = frame.reshape(msg.height, msg.step)
            frame = frame[:, :msg.width * 3]
            frame = frame.reshape(msg.height, msg.width, 3).copy()
        except Exception as e:
            self.get_logger().error(f'Image 변환 오류: {e}')
            return

        with self.lock:
            detections = [dict(x) for x in self.detections]

        self.draw_detections(frame, detections)
        ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return

        with self.lock:
            self.latest_jpeg = encoded.tobytes()
            self.last_image_time = time.monotonic()

    def detection_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
        with self.lock:
            self.detections = data
            self.last_detection_time = time.monotonic()

    def target_callback(self, msg):
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                return
        except json.JSONDecodeError:
            return
        with self.lock:
            self.nearest_target = data
            self.last_target_time = time.monotonic()

    def joint_callback(self, msg):
        with self.lock:
            self.joint_names = list(msg.name)
            self.joint_positions = list(msg.position)
            self.last_joint_time = time.monotonic()

    def draw_detections(self, frame, detections):
        colors = {'paper': (0, 0, 255), 'pet': (255, 0, 0), 'can': (0, 255, 0)}
        for det in detections:
            try:
                waste_type = str(det.get('type', 'unknown'))
                class_name = str(det.get('class', 'unknown'))
                confidence = float(det.get('confidence', 0.0))
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                cx, cy = int(det['center_x']), int(det['center_y'])
            except (KeyError, TypeError, ValueError):
                continue
            color = colors.get(waste_type, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            label = f'{waste_type.upper()} {class_name} {confidence:.2f}'
            cv2.putText(frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def topic_status(self, last_time, timeout=2.0):
        if last_time is None:
            return {'online': False, 'age': None}
        age_value = time.monotonic() - last_time
        return {'online': age_value <= timeout, 'age': age_value}

    def index(self):
        return HTML_PAGE

    def api_status(self):
        with self.lock:
            detections = [dict(x) for x in self.detections]
            target = dict(self.nearest_target) if self.nearest_target is not None else None
            names = list(self.joint_names)
            positions = list(self.joint_positions)
            ti = self.last_image_time
            td = self.last_detection_time
            tt = self.last_target_time
            tj = self.last_joint_time

        counts = {'paper': 0, 'pet': 0, 'can': 0}
        for det in detections:
            if det.get('type') in counts:
                counts[det['type']] += 1

        joints = []
        for name, position in zip(names, positions):
            joints.append({
                'name': name,
                'position_rad': position,
                'position_deg': math.degrees(position)
            })

        return jsonify({
            'counts': counts,
            'nearest_target': target,
            'detections': detections,
            'joints': joints,
            'topics': {
                'image_raw': self.topic_status(ti),
                'detections': self.topic_status(td),
                'nearest_target': self.topic_status(tt),
                'joint_states': self.topic_status(tj),
            }
        })

    def video_generator(self):
        while rclpy.ok():
            with self.lock:
                jpeg = self.latest_jpeg
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            time.sleep(0.03)

    def video_feed(self):
        return Response(self.video_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def run_flask(self):
        self.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)


def main(args=None):
    rclpy.init(args=args)
    node = WasteDashboardNode()
    flask_thread = threading.Thread(target=node.run_flask, daemon=True)
    flask_thread.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()