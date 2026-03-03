# import libraries
from deepdiff import DeepDiff
from engineering_notation import EngNumber
import csv
import os
import pathlib

def metadata_params(data, parameter=None, parameter_false=None, readable=True, write=False, out_folder_override=False):
    if hasattr(data, 'folder') and hasattr(data, 'stack_meta'):
        slash = slash_type(data.folder)
        stack_meta = data.stack_meta
        name = rf"{data.date}_{data.name}"
        out_folder = data.folder.replace('Raw', 'Processed') if out_folder_override else data.folder
        print(out_folder)
        file_stem = data.name
    elif isinstance(data, dict) and ('img1' in data or 'System' in data):
        slash = '/'
        stack_meta = {'img1': data} if 'System' in data else data
        name = 'stack'
        out_folder = '.'
        file_stem = 'stack'
    else:
        raise TypeError("metadata_params expects a ps.data object or a stack_meta dict")
    params = []
    if isinstance(parameter, str):
        params.append(parameter)
    if isinstance(parameter, list):
        params = parameter
    params_false = []
    if isinstance(parameter_false, str):
        params_false.append(parameter_false)
    if isinstance(parameter_false, list):
        params_false = parameter_false
    if 'stage' in params_false:
        params_false = [p for p in params_false if p != 'stage']
        params_false.extend(['r', 't', 'x', 'y', 'z'])
    prop_list = ['curr', 'accel', 'uc', 'dwell', 'wd', 'r', 't', 'x', 'y', 'z', 'hfw', 'px', 'py', 'average', 'integrate', 'interlacing', 'cntr', 'brtn']
    if parameter is None:
        params = prop_list
    if isinstance(params_false, list):
        params = [p for p in params if p not in params_false]
    if write:
        md_dir = rf"{out_folder.replace('Raw','Processed')}{slash}Metadata"
        os.makedirs(md_dir, exist_ok=True)
        with open(rf"{md_dir}{slash}{file_stem}_stack_meta_readable.txt", "w") as f:
            f.write(f'{name.center(42)}\n')
            f.write('_' * 42 + '\n')
            f.write('{:<15s} {:<15s} {:<15s}'.format('parameter', 'value', 'unit') + '\n')
            f.write('_' * 42 + '\n')
    if readable:
        print('\n', rf'{name.center(42)}')
        print('_' * 42)
        print('{:<15s} {:<15s} {:<15s}'.format('parameter', 'value', 'unit'))
        print('_' * 42)
    for prop in params:
        if prop == 'curr':
            k, unit = ['EBeam', 'BeamCurrent'], 'A'
        elif prop == 'accel':
            k, unit = ['Beam', 'HV'], 'V'
        elif prop == 'uc':
            k, unit = ['EBeam', 'BeamMode'], ''
        elif prop == 'dwell':
            k, unit = ['EScan', 'Dwell'], 's'
        elif prop == 'wd':
            k, unit = ['Stage', 'WorkingDistance'], 'm'
        elif prop == 'hfw':
            k, unit = ['EScan', 'HorFieldsize'], 'm'
        elif prop == 'px':
            k, unit = ['Image', 'ResolutionX'], 'pixels'
        elif prop == 'py':
            k, unit = ['Image', 'ResolutionY'], 'pixels'
        elif prop == 'average':
            k, unit = ['Image', 'Average'], 'frames'
        elif prop == 'integrate':
            k, unit = ['Image', 'Integrate'], 'frames'
        elif prop == 'interlacing':
            k, unit = ['EScan', 'ScanInterlacing'], 'lines'
        elif prop in ['step', 'range']:
            k, unit = ['TLD', 'Mirror'], 'V'
        elif prop == 'r':
            k, unit = ['Stage', 'StageR'], 'rad'
        elif prop == 't':
            k, unit = ['Stage', 'StageT'], 'rad'
        elif prop == 'brtn':
            k, unit = ['TLD', 'Brightness'], '%'
        elif prop == 'cntr':
            k, unit = ['TLD', 'Contrast'], '%'
        elif prop in ['x', 'y', 'z']:
            k, unit = ['Stage', rf'Stage{prop.upper()}'], 'm'
        else:
            continue
        if 'img1' in stack_meta:
            value = stack_meta['img1']
        else:
            value = stack_meta
        for key in k:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                break
        v = str(EngNumber(value)) if isinstance(value, (float, int)) else value
        if write:
            md_dir = rf"{out_folder.replace('Raw','Processed')}{slash}Metadata"
            with open(rf"{md_dir}{slash}{file_stem}_stack_meta_readable.txt", "a") as f:
                f.write('{:<15s} {:<15s} {:<15s}\n'.format(prop, str(v), unit))
        if readable:
            print('{:<15s} {:<15s} {:<15s}'.format(prop, str(v), unit))
        if not readable and parameter is not None and isinstance(parameter, str):
            return k, value

def compare_params(meta_1, meta_2, readable=True, condition_true: list = None):
    if 'img1' in meta_1.keys():
        meta_1 = meta_1['img1']
    if 'img1' in meta_2.keys():
        meta_2 = meta_2['img1']
    diff_out = DeepDiff(meta_1, meta_2, exclude_paths=["root['PrivateFEI']", "root['PrivateFei']", "root['User']", "root['System']", "root['Processing']", "root['GIS']"])
    diff = {}
    if 'values_changed' in diff_out:
        for page in diff_out['values_changed']:
            diff[page.split("'")[1]] = {}
        for page in diff_out['values_changed']:
            diff[page.split("'")[1]][page.split("'")[3]] = diff_out['values_changed'][page]
    return diff

def wd_check(data, readable=True, distance=0.0040):
    k, v = metadata_params(data, 'wd', readable=False)
    if round(v, 4) == distance:
        if readable is False:
            return True
    else:
        if readable is True:
            print('\t\tWARNING !', '\t\t', 'Working distance is \t', v * 1000, ' mm')
        if readable is False:
            return False
          
def slash_type(path):
    if type(pathlib.Path(path)) is pathlib.WindowsPath:
        return '\\'
    return '/'
