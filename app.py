from flask import Flask, jsonify, request, Response
from flask.helpers import send_from_directory
from flask_cors import CORS, cross_origin
from google.cloud import speech
import speech_recognition as sr
import requests
import threading
import os
import io
import heapq
import math
import time

app = Flask(__name__)
CORS(app)

# Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'robotics-project-431719-402cadd1538d.json'

# Audio recording parameters
RATE = 16000

# Global variable to store the computed path
computed_path = None

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=RATE)

    def __enter__(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        return self

    def __exit__(self, type, value, traceback):
        pass

    def generator(self):
        while True:
            with self.microphone as source:
                audio = self.recognizer.listen(source)
                yield audio.get_raw_data()

def listen_print_loop(responses):
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        yield f"data: {transcript}\n\n"

@app.route('/stream/transcription', methods=['GET'])
@cross_origin()
def stream_transcription():
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    def generate_responses():
        with MicrophoneStream() as stream:
            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_generator)
            responses = client.streaming_recognize(streaming_config, requests)
            return listen_print_loop(responses)

    return Response(generate_responses(), mimetype='text/event-stream')

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
