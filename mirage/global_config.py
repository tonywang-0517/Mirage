#!/usr/bin/env python3
# coding: utf-8

from easydict import EasyDict

Config = EasyDict({
    "control_delay": True,
    "randomize_body_mass": True,
    "randomize_joint_parameters": True,
    "randomize_body_com": True,
    "randomize_actuator_gains": True,
    "randomize_rfi_lim": False,
    "random_push_robot": True,
    "randomize_physics_scene_gravity": True,
})