#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:56:26 2021

@author: E. Syniawa
"""
import ANNarchy as ann
import numpy as np
from os import path, makedirs


class Pop_Monitor(object):
    def __init__(self, populations, samplingrate=2.0):
        self.populations = populations
        self.monitors = []
        for pop in populations:
            self.monitors.append(ann.Monitor(pop, 'r', period=samplingrate, start=False))
    def start(self):
        for monitor in self.monitors:
            monitor.start()
    def stop(self):
        for monitor in self.monitors:
            monitor.pause()
    def resume(self):
        for monitor in self.monitors:
            monitor.resume()
    def get(self):
        res= {}
        for monitor in self.monitors:
            res[monitor.object.name] = monitor.get('r')
        return res
    def save(self, folder):
        if not path.exists(folder):
            makedirs(folder)

        for monitor in self.monitors:
            rec = monitor.get('r')
            np.save(folder + 'r' + monitor.object.name, rec)
    def load(self, folder):
        monitor_dict = {}
        for monitor in self.monitors:
            monitor_dict['r' + monitor.object.name] = np.load(folder + 'r' + monitor.object.name + '.npy')
        return monitor_dict


class Con_Monitor(object):
    def __init__(self, connections):
        self.connections = connections
        self.weight_monitors = {}
        for con in connections:
            self.weight_monitors[con.name] = []
    def extract_weights(self):
        for con in self.connections:
            weights = np.array([dendrite.w for dendrite in con])
            self.weight_monitors[con.name].append(weights)
    def save_cons(self, folder):
        if not path.exists(folder):
            makedirs(folder)

        for con in self.connections:
            np.save(folder + 'w' + con.name, self.weight_monitors[con.name])
    def load_cons(self, folder):
        con_dict = {}
        for con in self.connections:
            con_dict['w' + con.name] = np.load(folder + 'w' + con.name + '.npy')
    def reset(self):
        for con in self.connections:
            self.weight_monitors[con.name] = []