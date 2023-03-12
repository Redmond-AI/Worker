from queue import Queue
from threading import Lock, Thread
import json
import os
from webui_node import image_generator
import socketio
from socketio.exceptions import ConnectionRefusedError
import uuid
from os.path import join, abspath, dirname


CONFIG_PATH = join(dirname(abspath(__file__)), "cfg", "basic.json")

sio = socketio.Client(reconnection=True, reconnection_attempts=0, reconnection_delay=0, reconnection_delay_max=0, randomization_factor=0)

NODE_KEY = 'Cj4UyVUJV8GmJ89vy5SKAS7d'

ex_uuid = str(uuid.uuid4())

# CONFIG_PATH = "/storage/node/cfg/basic.json"

with open(CONFIG_PATH) as f:
    config = json.load(f)

# Create the request queue and image queue
request_queue = Queue()
image_queue = Queue()
command_queue = Queue()
queue_lock = Lock()
global first_time
first_time = True

@sio.event
def connect():
    print("Connected...")
    global first_time
    if first_time is True:
        print("First Time...")
        response = command_queue.get()
        print("Obtained!")
        if response['status'] != 'ready':
            raise Exception("Unexpected status on init")
        first_time = False
    else:
        print("Not first time..")

    available_models = []
    for entry in config['models']:
        available_models.append(entry['alias'])
    print("joining!")
    sio.emit('join', data={'room': 't2i', 'instance_id': ex_uuid})
    print("Joined!")

    data_to_update = {
        "INSTANCE_ID": ex_uuid,
        "NEW_RECORDS": {
            "MODELS": available_models,
            "STATUS": "ready",
            "TASKS": 0
        }
    }
    print("Updaiting records..")
    sio.emit('update_records', data=data_to_update)
    print("Updated.")

@sio.event
def disconnect():
    print("Disconnected")
    sio.connect(config['server'], auth={'KEY': NODE_KEY}, wait_timeout=10)

@sio.on('CONNECTION_STATUS')
def on_connection_status(data):
    if data == 'SUCCESS' is False:
        raise ConnectionRefusedError('FAILED TO AUTH')
    else:
        print("AUTH SUCCESS")

@sio.on('status')
def on_message(data):
    print('I received a status!', data)

@sio.on('*')
def catch_all(event, data):
    print("Catched some event:", event)
    print("data:", data)
    
@sio.on('task')
def on_task(data):
    print(data)
    queue_pointer = data['QUEUE_POINTER']
    job_key = data['JOB_KEY']
    parameters = data['PARAMETERS']
    with queue_lock:
        r = {
            "prompt": str(parameters['prompt']),
            "negative_prompt": str(parameters['negative_prompt']) if 'negative_prompt' in parameters else None,
            "model": str(parameters['model']),
            "vae": str(parameters['vae']),
            "steps": int(parameters['steps']),
            "width": int(parameters['width']),
            "height": int(parameters['height']),
            "cfg": float(parameters['cfg']),
            "seed": int(parameters['seed']),
            "scheduler": str(parameters['scheduler']),
        }
        request_queue.put(r)
    response_queue = image_queue.get()
    response = {
        "JOB_KEY": job_key,
        "QUEUE_POINTER": queue_pointer,
        "DATA": response_queue
    }
    print("I guess i posted it")
    if sio.connected is False:
        pass
    elif sio.connected is True:
        sio.emit('post_task', response)

# Image Generator Engine
image_generator_thread = Thread(
    target=image_generator, 
    args=(
        config, 
        request_queue,
        image_queue,
        command_queue,
        )
    )
image_generator_thread.start()

print("Connecting...")
sio.connect(config['server'], auth={'KEY': NODE_KEY}, wait_timeout=5)

