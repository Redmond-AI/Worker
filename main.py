from queue import Queue
from threading import Lock, Thread
import json
import os
from threads.base import image_generator
import socketio
from socketio.exceptions import ConnectionRefusedError
import uuid
sio = socketio.Client()

NODE_KEY = 'Cj4UyVUJV8GmJ89vy5SKAS7d'

ex_uuid = str(uuid.uuid4())

CONFIG_PATH = "cfg/basic.json"

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        config = json.load(f)

# Create the request queue and image queue
request_queue = Queue()
image_queue = Queue()
command_queue = Queue()
queue_lock = Lock()

@sio.event
def connect():
    print("Connected...")
    response = command_queue.get()
    if response['status'] != 'ready':
        raise Exception("Unexpected status on init")

    available_models = []
    for entry in config['models']:
        available_models.append(entry['alias'])

    data_to_update = {
        "INSTANCE_ID": ex_uuid,
        "NEW_RECORDS": {
            "MODELS": available_models,
            "STATUS": "ready",
            "TASKS": 0
        }
    }

    sio.emit('update_records', data=data_to_update)

@sio.event
def disconnect():
    print("Disconnected")

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

sio.connect(config['server'], auth={'KEY': NODE_KEY}, wait_timeout=30)
sio.emit('join', data={'room': 't2i', 'instance_id': ex_uuid})

