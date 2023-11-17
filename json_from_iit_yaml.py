#!/usr/bin/env python3
# Author: Jonathan Külz
# Date: 13.11.23
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

from timor.utilities.spatial import rotX
import yaml as yaml_lib

import numpy as np
import pinocchio as pin
from timor import AtomicModule, Body, Connector, Geometry, Joint, ModuleHeader, ModulesDB, Transformation
from timor.utilities import spatial

head = Path(__file__).parent
package = head.parent
meshes = head / 'modular/meshes/alberobotics/'
module_data = head / 'yaml'


def dh_ext(a_d: float, alpha_d: float, n_d: float, p_d: float, delta: float) -> Transformation:
    """
    Transforms extended DH-parameters to a 4x4 homogeneous transformation.

    Takes parameters from distal link i-1 and proximal link i.
    Formula taken from "Automatic Centralized Controller Design for Modular and Reconfigurable Robot Manipulators",
    A. Giusti and M. Althoff, 2015, eq. (2).

    IIT uses p instead of -p, so the first translation is positive here.
    """
    return Transformation.from_translation([0, 0, p_d]) \
        @ Transformation.from_translation([a_d, 0, 0]) \
        @ Transformation(spatial.rotX(alpha_d)) \
        @ Transformation.from_translation([0, 0, n_d]) \
        @ Transformation(spatial.rotZ(delta))


def dh_ext_from_dict(d: Dict) -> Transformation:
    """Wrapper for dh_ext that takes a dictionary and maps the different names to the right argument."""
    keys = list(d.keys())
    args = {'a_d': None, 'alpha_d': None, 'n_d': None, 'p_d': None}
    for arg in args:
        name = arg.split('_')[0] + '_'
        matches = [key for key in keys if key.startswith(name)]
        assert len(matches) == 1, f"Expected exactly one match for {name}, got {matches}"
        args[arg] = d.pop(matches[0])
    if 'delta_dl' in keys:
        args['delta'] = d.pop('delta_dl')
    elif 'delta_pl' in keys:
        args['delta'] = d.pop('delta_pl')
    elif 'delta_l_in' in keys and 'delta_l_out' in keys:
        args['delta'] = d.pop('delta_l_in') + d.pop('delta_l_out')
    return dh_ext(**args)


def make_geometry(info: Optional[Dict]) -> Optional[Geometry.Geometry]:
    """Creates a geometry objects from URDF-like data."""
    if info is None:
        return None

    if isinstance(info, list):
        assert len(info) == 1
        info = info[0]

    geometry_type = info.pop('type')
    parameters = info.pop('parameters')
    pose_info = info.pop('pose', None)
    assert len(info) == 0, f'Unknown keys: {info.keys()}'
    t = np.array([pose_info['x'], pose_info['y'], pose_info['z']])
    r = spatial.euler2mat(np.array([pose_info['roll'], pose_info['pitch'], pose_info['yaw']]), 'xyz')
    pose = Transformation.from_roto_translation(r, t)

    return Geometry.Geometry.from_json_data({
        'type': geometry_type,
        'parameters': parameters,
        'pose': pose,
    }, package_dir=package)


def make_module(d: Dict) -> AtomicModule:
    """Create a module from a yaml file."""
    if d['type'] == 'tool_exchanger':
        raise NotImplementedError('Tool exchanger is not supported')
    header = d.pop('header')
    for tag in ('author', 'affiliation', 'email'):
        if isinstance(header[tag], str):
            header[tag] = [header[tag]]
    for tag in ('disabled', 'type', 'family', 'label'):
        header.pop(tag, None)
    header = ModuleHeader(**header)

    d.pop('gazebo', None)
    d.pop('kinematics_convention', None)
    d.pop('urdf', None)
    d.pop('CentAcESC', None)
    d.pop('LpESC', None)

    body_arguments = dict()
    collisions = d.pop('collision')
    visuals = d.pop('visual', None)
    for body_name in collisions:
        collision = make_geometry(collisions[body_name])
        visual = make_geometry(visuals.get(body_name, None))
        body_arguments[body_name] = dict(body_id=body_name, collision=collision, visual=visual)

    dynamics = d.pop('dynamics')
    rotor_inertia = 0
    for body_name in dynamics:
        inert = dynamics[body_name].pop('inertia_tensor')
        ixx, ixy, ixz, iyy, iyz, izz = inert['I_xx'], inert['I_xy'], inert['I_xz'], inert['I_yy'], inert['I_yz'], inert[
            'I_zz']
        inertial_data = pin.Inertia(
            mass=dynamics[body_name]['mass'],
            lever=np.array(
                [dynamics[body_name]['CoM']['x'], dynamics[body_name]['CoM']['y'], dynamics[body_name]['CoM']['z']]),
            inertia=np.array([
                [ixx, ixy, ixz],
                [ixy, iyy, iyz],
                [ixz, iyz, izz],]))
        if body_name.endswith('_fast'):
            # The "fast" bodies contain information about the "fast-moving-side" of a joint, so the rotor before the
            # gear. We handle it slightly hacky by just adding it to the corresponding rigid body and setting the
            # rotor_inertia to the z-component of the inertia tensor.
            body_arguments[body_name[:-5]]['inertia'] += inertial_data
            rotor_inertia = inertial_data.inertia[2, 2]
        else:
            body_arguments[body_name]['inertia'] = inertial_data

    kinematics = d.pop('kinematics')
    if kinematics['joint'] is None:
        assert d.pop('type') == 'link', 'Only links can have no joint'
        has_joint = False
    else:
        assert d.pop('type') in ('gripper', 'joint'), 'Only joints and grippers can have a joint'
        has_joint = True
        t_proximal = dh_ext_from_dict(kinematics['joint']['proximal'])
        t_distal = dh_ext_from_dict(kinematics['joint']['distal'])

    t_link = dh_ext_from_dict(kinematics['link'])

    if d.pop('size_in', None) is not None:
        raise ValueError('size_in is not supported')
    if d.pop('size_out', None) is not None:
        raise ValueError('size_out is not supported')
    c_type = d.pop('flange_size')

    if has_joint:
        joint_data = d.pop('actuator_data')
        assert joint_data['zero_offset'] == 0, 'Zero offset is not supported'
        assert len(body_arguments) == 2, "Only two bodies are allowed for links with joints"
        c_in = Connector(header.ID + '_in', Transformation(rotX(np.pi)), gender='f', connector_type=c_type)
        proximal_body = Body(**body_arguments['body_1'], connectors=(c_in,))
        c_out = Connector(header.ID + '_out', t_distal, gender='f', connector_type=c_type)
        distal_body = Body(**body_arguments['body_2'], connectors=(c_out,))
        bodies = (proximal_body, distal_body)
        joints = (Joint(header.ID + '_joint', joint_data['type'], proximal_body, distal_body,
                        q_limits=[joint_data['lower_limit'], joint_data['upper_limit']],
                        torque_limit=joint_data['effort'],
                        motor_inertia=rotor_inertia,
                        velocity_limit=joint_data['velocity'], parent2joint=t_proximal,
                        joint2child=Transformation.neutral(), gear_ratio=joint_data['gear_ratio']),)
    else:
        connectors = (
            Connector(header.ID + '_in', Transformation(rotX(np.pi)), gender='f', connector_type=c_type),
            Connector(header.ID + '_out', t_link, gender='m', connector_type=c_type)
        )
        assert len(body_arguments) == 1, "Only one body is allowed for links without joints"
        body_name = list(body_arguments.keys())[0]
        bodies = (Body(**body_arguments[body_name], connectors=connectors),)
        joints = ()

    assert len(d) == 0, f'Unknown keys: {d.keys()}'

    return AtomicModule(header=header, bodies=bodies, joints=joints)


def main():
    db = ModulesDB(name='modular_resources')  # The name must match the repository name
    for file in module_data.rglob('*.yaml'):
        if file.name in ('master_cube.yaml', 'template.yaml', 'module_joint_elbow_ORANGE.yaml',
                         'module_tool_exchanger.yaml', 'module_tool_exchanger_heavy.yaml'):
            continue
        with file.open('r') as f:
            content = yaml_lib.safe_load(f)
        new_module = make_module(content)
        new_module.to_json_file(head.joinpath(f'cobra_json/{new_module.name}.json'), handle_missing_assets='symlink')
        db.add(new_module)
    db.debug_visualization()
    db.to_json_file(head.joinpath('modules.json'))
    input('Press enter to quit...')


if __name__ == '__main__':
    main()
