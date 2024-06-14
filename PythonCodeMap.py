'''
    This file is used to capture the stack trace of user code in a separate thread
    and save it in a file.
    Run this script in debug console.
        
'''

import threading
import sys
import os
import pdb
import json

file_name = __file__
print(file_name)

def is_user_code_file(file_path):
    return file_path.startswith(os.getcwd())

def get_paused_thread_stack_traces():
    stack_traces = {}
    for thread_id, frame in sys._current_frames().items():
        stack_trace = []
        while frame:
            file_path = frame.f_code.co_filename
            if is_user_code_file(file_path) and not file_path == file_name:
                stack_trace.append(f"{file_path}:{frame.f_lineno} - {frame.f_code.co_name}")
            frame = frame.f_back
        if stack_trace:
            stack_traces[thread_id] = stack_trace
    return stack_traces

def save_user_code_thread_stack_trace(previous_trace=""):
    print("Saving user code thread stack traces...")
    paused_thread_stack_traces = get_paused_thread_stack_traces()
    
    # convert paused_thread_stack_traces to json string and compare with previous json string
    # if matching with previous string then do not print. TO AVOID DUPLICATE PRINT
    current_trace = json.dumps(paused_thread_stack_traces)
    print(f"json data: {current_trace}")
    if previous_trace == current_trace:
        print("No changes in user code thread stack traces.")
        return current_trace
    previous_trace = current_trace

    for thread_id, stack_trace in paused_thread_stack_traces.items():
        print(f"Thread ID: {thread_id}")
        for frame in stack_trace:
            print(f"  {frame}")

        file_name = f"zTrace_user_code_thread_stack_trace.txt"
        with open(file_name, 'a') as file:
            file.write(f"\n\n################################################# - ThreadId - {thread_id}\n\n")
            for frame in stack_trace:
                file.write(f"{frame}\n")
            file.write(f"\n\n###########################################################################\n\n")    

    return current_trace

import time 
def start_trace_capture():
    previous_trace = ""
    while True:
        time.sleep(1)
        previous_trace = save_user_code_thread_stack_trace(previous_trace="")

def debug_function():
    # Create a new thread for the debug_function
    debug_thread = threading.Thread(target=start_trace_capture)
    # Start the debug thread
    debug_thread.start()


# def initialize_debugger():
#     import ptvsd
#     ptvsd.enable_attach()
#     #ptvsd.wait_for_attach()
#     ptvsd.break_into_debugger()

# initialize_debugger()



# import traceback
# import threading
# import os
# import sys

# def is_user_code_frame(frame):
#     return frame #.f_code.co_filename.startswith(os.getcwd())

# def save_user_code_thread_stack_trace():
#     stack_traces = {}
#     for thread in threading.enumerate():
#         print(thread.name, thread.ident, thread.isDaemon()) 
#         stack_trace = ''.join(traceback.format_stack(f=None, limit=None))
#         stack_frames = traceback.extract_stack(f=None, limit=None)
#         user_code_frames = [frame for frame in stack_frames if is_user_code_frame(frame)]
#         for user_code_frame in user_code_frames:
#             print(user_code_frame)
#         print("================================================")
#         if user_code_frames:
#             user_code_stack_trace = ''.join(traceback.format_list(user_code_frames))
#             stack_traces[thread.ident] = user_code_stack_trace

#     for thread_id, stack_trace in stack_traces.items():
#         file_name = f"user_code_thread_{thread_id}_stack_trace.txt"
#         with open(file_name, 'w') as file:
#             file.write(stack_trace)

# save_user_code_thread_stack_trace()







