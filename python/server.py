import socket
import json

import matplotlib.pyplot as plt

from plotting import *
from copy import deepcopy

HOST = '127.0.0.1'
PORT = 8888
BUFFER_SIZE = 1024 * 1024 * 1024


def process_message(msg):
    name = None
    method = "plot"
    h = None
    dt = 0.0
    E = 0.0
    L = 1.0

    data = json.loads(msg.decode())

    print(data["header"])

    if "name" in data["header"]:
        name = data["name"]
    if "method" in data["header"]:
        method = data["method"]
    if "L" in data["header"]:
        L = data["L"]
    if "h" in data["header"]:
        h = data["h"]
    if "dt" in data["header"]:
        dt = data["dt"]

    if "t" in data["header"]:
        time = data["t"]
        t = np.arange(0.0, time, dt)
    elif "times" in data["header"]:
        t = data["times"]

    if "E" in data["header"]:
        E = data["E"]

    title = f"{name}\nE = {E}"
    if h is not None:
        title += f" h = {h}"

    if "x" in data["header"] and method == "simplePlot":
        x = data["x"]
        plt.figure()
        plt.plot(x)
        plt.show()

    # TODO: cycle by "results"
    if name == "Oscillator mLCE(h, E)":
        plot_mLCE_diagram(data["mLCEs"], data["E_vals"], data["h_steps"], name)
        
    if name == "Divergence Degree":
        traj1 = data["traj1"]
        traj2 = data["traj2"]
        
        # tr1 = deepcopy(traj1[:2000])
        
        x1, y1 = [i[0] for i in traj1], [i[1] for i in traj1]
        x2, y2 = [i[0] for i in traj2], [i[1] for i in traj2]
        
        plt.plot(x1, y1)
        plt.plot(x2, y2)
        plt.show()
        return

    if "results" in data["header"]:
        if "derivatives" in data["results"] and method == "plot":
            x = data["results"]["x"]
            derivatives = data["results"]["derivatives"]
            plot_preisach_derivative(t, x, derivatives, "")
        # for res in data["results"]:
        if "x" in data["results"] and method == "plot":
            x = data["results"]["x"]
            plot_x(x, t, title)
        if "v" in data["results"] and method == "plot":
            v = data["results"]["v"]
            plot_v(v, t, title)
        if "x" in data["results"] and "v" in data["results"] and method == "plot":
            x = data["results"]["x"]
            v = data["results"]["v"]
            plot_phase_portrait(x, v, title)
    if "loop" in data["header"] and method == "plot":
        inputs = data["loop"]["inputs"]
        outputs = data["loop"]["outputs"]
        plot_hysteresis_loop(inputs, outputs, title)
    if "anim" in data["header"]:
        sim = animate_preisach_plane(data["anim"]["in"], data["anim"]["x"], data["anim"]["y"], data["anim"]["out"], L)
        if data["anim"]["save"]:
            import matplotlib.animation as animation
            writer = animation.FFMpegWriter(fps=30, metadata=dict(), bitrate=1800)
            print("Saving hysteresis animation...")
            sim.save(filename='preisach_sim.mp4', writer=writer)
            print("Animation saved!")


class Server:
    def __init__(self, host=HOST, port=PORT):
        self.port = port
        self.host = host

    def start(self):
        self._init()
        print("Server Started")

        conn, addr = self.socket.accept()
        with conn:
            print(f"Connected by: {addr}")
            while True:
                try:
                    msg = conn.recv(BUFFER_SIZE)
                    if not msg:
                        break
                    process_message(msg)
                except ConnectionResetError:
                    print("Connection aborted")
                    break

    def _init(self):
        # self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((HOST, PORT))
        self.socket.listen()

    def __del__(self):
        self.socket.close()
