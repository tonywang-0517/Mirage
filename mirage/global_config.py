#!/usr/bin/env python3
# coding: utf-8

from easydict import EasyDict

# Config = EasyDict({
#     "control_delay": True, #ok
#     "randomize_body_mass": True, #ok
#     "randomize_joint_parameters": True, #ok
#     "randomize_body_com": False, #bug
#     "randomize_actuator_gains": True, #ok
#     "randomize_rfi_lim": False,#bug
#     "random_push_robot": False, #not necessary
#     "randomize_physics_scene_gravity": True, #ok
#     "enabled_self_collisions": True
# })

Config = EasyDict({
    "control_delay": False, #ok
    "randomize_body_mass": False, #ok
    "randomize_joint_parameters": False, #ok
    "randomize_body_com": False, #bug
    "randomize_actuator_gains": False, #ok
    "randomize_rfi_lim": False,#bug
    "random_push_robot": False, #not necessary
    "randomize_physics_scene_gravity": False, #ok
    "enabled_self_collisions": True,
    "use_delta": False,
    "freeze_delta": False
})

#python mirage/target_domain_data_collect.py +exp=full_body_tracker/transformer_flat_terrain +robot=g1 +simulator=genesis +motion_file=data/motions/g1_walk.npy +experiment_name=g1_walk_DR_action ++headless=False ++num_envs=1 +checkpoint=results/g1_walk_DR_action/last.ckpt