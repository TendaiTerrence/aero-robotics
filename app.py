from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
from google.cloud import speech
import os
import requests
import time
import heapq
import math

app = Flask(__name__)
CORS(app)

# Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/etc/secrets/robotics-project-431719-402cadd1538d.json'

# Global variable to store the computed path
computed_path = None

@app.route('/upload_audio', methods=['POST'])
@cross_origin()
def upload_audio():
    file = request.files['file']
    audio_data = file.read()

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        transcript = result.alternatives[0].transcript
        print(f"Transcript: {transcript}")
    
    return jsonify({"transcript": transcript}), 200

@app.route('/record/send', methods=['POST'])
@cross_origin()
def record_send():
    commands = request.json.get('commands', [])
    if not commands:
        return jsonify(status='error', message='No commands received'), 400

    for command in commands:
        send_command_to_ev3(command)

    return jsonify(status='command sent')

@app.route('/api/optimize-path', methods=['POST'])
@cross_origin()
def optimize_path():
    global computed_path
    grid_data = request.json

    grid = grid_data['grid']
    start = tuple(grid_data['start'])
    goal = tuple(grid_data['goal'])

    start_time = time.time()  # Start timing

    computed_path = astar(grid, start, goal)

    end_time = time.time()  # End timing
    time_taken = end_time - start_time  # Calculate the time taken

    if computed_path is None:
        return jsonify({"error": "No path found"}), 404
    else:
        # Send the computed path to the EV3 robot
        # send_path_to_ev3(computed_path)
        return jsonify({"path": computed_path}), 200

@app.route('/get-path', methods=['GET'])
@cross_origin()
def get_path():
    global computed_path

    if computed_path is None:
        return jsonify({"error": "No path computed"}), 404
    else:
        return jsonify({"path": computed_path}), 200
    
@app.route('/command', methods=['POST'])
@cross_origin()
def handle_command():
    command = request.json.get('command', '')
    print(f"Received command: {command}")
    send_command_to_ev3(command)
    return jsonify(status='command received')

def send_command_to_ev3(command):
    ev3_ip = "192.168.0.1"  
    url = f"http://{ev3_ip}:5000/command"
    data = {"command": command}
    try:
        response = requests.post(url, json=data)
        print(f"Sent command to EV3: {data}")
        print(f"EV3 response status: {response.status_code}, text: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send command to EV3: {e}")

def send_path_to_ev3(path):
    """Convert the path into a single command and send it to the EV3 robot."""
    ev3_ip = "192.168.0.1"  
    url = f"http://{ev3_ip}:5000/path"
    try:
        response = requests.post(url, json={"path": path})
        print(f"Sent path to EV3: {path}")
        print(f"EV3 response status: {response.status_code}, text: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send path to EV3: {e}")

# A* Pathfinding Implementation
class Node:
    def __init__(self, position, parent=None):
        self.position = position  # (row, col)
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Estimated cost from current node to goal
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

def heuristic(node_position, goal_position):
    return math.sqrt((node_position[0] - goal_position[0]) ** 2 + (node_position[1] - goal_position[1]) ** 2)

def astar(grid, start, goal):
    start_node = Node(start)
    goal_node = Node(goal)
    open_list = []
    closed_list = set()
    heapq.heappush(open_list, start_node)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)
        
        if current_node == goal_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]
        
        for direction in directions:
            new_position = (current_node.position[0] + direction[0], current_node.position[1] + direction[1])
            
            if 0 <= new_position[0] < len(grid) and 0 <= new_position[1] < len(grid[0]):
                if grid[new_position[0]][new_position[1]] == 1:
                    continue
                
                child_node = Node(new_position, current_node)
                
                if child_node.position in closed_list:
                    continue
                
                child_node.g = current_node.g + 1
                child_node.h = heuristic(child_node.position, goal_node.position)
                child_node.f = child_node.g + child_node.h
                
                if any(open_node for open_node in open_list if child_node == open_node and child_node.g > open_node.g):
                    continue
                
                heapq.heappush(open_list, child_node)
    
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
