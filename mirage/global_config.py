#!/usr/bin/env python3
# coding: utf-8

from easydict import EasyDict

Config = EasyDict({
    "control_delay": False, #ok
    "randomize_body_mass": True, #ok
    "randomize_joint_parameters": True, #ok
    "randomize_body_com": False, #bug
    "randomize_actuator_gains": True, #ok
    "randomize_rfi_lim": False,#bug
    "random_push_robot": False, #not necessary
    "randomize_physics_scene_gravity": True, #ok
    "enabled_self_collisions": False
})

