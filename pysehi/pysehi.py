# Import libraries
import os
import glob
import cv2
import tifffile as tf
import regex
import json
import numpy as np
import scipy.ndimage as scnd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.path import Path
import matplotlib.cm as cm
from matplotlib import colors
from cv2 import matchTemplate as match_template
from cv2 import minMaxLoc as min_max_loc
from cv2 import TM_CCOEFF_NORMED
from skimage import transform
from skimage.registration import phase_cross_correlation as pcc
from skimage import img_as_ubyte
from skimage import img_as_float
import read_roi
import output
import smooth
import metadata
import pathlib
import pandas as pd
import collections

def folder_finder(raw_folder):
    inlens = os.path.join(raw_folder, "InLens")
    esb = os.path.join(raw_folder, "ESB")
    inlens_files = sorted(glob.glob(os.path.join(inlens, "*.tif*")))
    esb_files = sorted(glob.glob(os.path.join(esb, "*.tif*")))
    return inlens_files, esb_files

def process_files(files: str or dict, AC: bool = True, condition_true: list = None, condition_false: list = None, register=True, custom_name=None, colour_thirds=None, overview_img: str = None, is_zeiss: bool = False):
    if type(files) is str:
        files_pl = pathlib.Path(files)
        init_date = None
        for part in files_pl.parts:
            m = regex.search(r"(\d{6}[\w-]*)", part)
            if m:
                init_date = m.group(1)
                break
        if init_date is None:
            print(r"Date missing! Input a path to raw files in the format '...\material\YYMmetadataD\Raw\...\...\data_folder')")
        elif 'Raw' not in files_pl.parts:
            print(r"Input a path to raw files in the format '...\material\YYMmetadataD\Raw\...\...\data_folder')")
        else:
            data_files = list_files(files, condition_true, condition_false, custom_name=custom_name, is_zeiss=is_zeiss)
            print(rf'found {len(data_files)} data folders for processing')
    elif type(files) is dict:
        data_files = files
    else:
        print("Unsupported files input")
        return
    for name in data_files:
        root = data_files[name]['Raw_path']
        leaf = os.path.basename(root).lower()
        if leaf in ('esb', 'inlens'):
            continue
        if os.path.exists(root.replace('Raw', 'Processed')) is True:
            print(r'already processed files associated with Raw_path')
        else:
            if '_R' not in root:
                print(rf'loading......{root}')
                data_files[name]['Processed_path'] = root.replace('Raw', 'Processed')
                dat = data(root, AC=AC, reg=register, is_zeiss=is_zeiss)
                dat.save_data(reg=register)
                data_files[name]['stack_meta'] = dat.stack_meta
                metadata.metadata_params(dat, write=True, out_folder_override=True)
                print(rf"Processed!......{root.replace('Raw', 'Processed')}")
            else:
                if AC is False:
                    print(rf'Loading......{root}')
                    data_files[name]['Processed_path'] = root.replace('Raw', 'Processed')
                    dat = data(root, AC=AC, reg=register, is_zeiss=is_zeiss)
                    dat.save_data(reg=register)
                    data_files[name]['stack_meta'] = dat.stack_meta
                    metadata.metadata_params(dat, write=True, out_folder_override=True)
                    print(rf"Processed!......{root.replace('Raw', 'Processed')}")
                else:
                    print(rf'AC is True, discounted......{root}')
    if isinstance(files, str):
        data_pro_path = files.replace('Raw', 'Processed')
    else:
        raw_roots = [v['Raw_path'] for v in data_files.values()]
        common_raw = os.path.commonpath(raw_roots)
        data_pro_path = common_raw.replace('Raw', 'Processed')
    output.summary_excel(data_pro_path, condition_true=condition_true, condition_false=condition_false, custom_name=custom_name, is_zeiss=is_zeiss)
    print("Excel summary sheet")
    if overview_img is not None:
        output.location_mosaic(data_pro_path, overview_img, condition_false)

def list_files(path_to_files, date: int = None, condition_true: list = None, condition_false: list = None,
               load_data=False, uint8=False, custom_name=None, is_zeiss: bool = False):
    data_files = {}
    for root, dirs, file_list in os.walk(path_to_files):
        root_pl = pathlib.Path(root)
        found_date = None
        for part in root_pl.parts:
            m = regex.search(r"(\d{6}[\w-]*)", part)
            if m:
                found_date = m.group(1)
                break
        if date is not None and not any(regex.search(rf"{date}", p) for p in root_pl.parts):
            continue
        if condition_true is not None and not any(c in root for c in condition_true):
            continue
        if condition_false is not None and any(c in root for c in condition_false):
            continue

        if 'Raw' in path_to_files:
            tail = os.path.basename(root).lower()
            if tail in ('inlens', 'esb'):
                continue

            inlens_dir = os.path.join(root, 'InLens')
            esb_dir = os.path.join(root, 'ESB')
            dual_ok = os.path.isdir(inlens_dir) and os.path.isdir(esb_dir)
            file_str = custom_name if custom_name is not None else 'TLD_Mirror'
            legacy_ok = any('Log.csv' in f for f in file_list) or any(file_str in f for f in file_list)

            if not (dual_ok or legacy_ok):
                continue
            if found_date is None:
                continue

            date_token = found_date
            name = os.path.split(root)[1]
            if root.find(date_token) > root.find('Raw'):
                if isinstance(root_pl, pathlib.WindowsPath):
                    try:
                        material = root.split(rf'\{date_token}')[0].split('Raw\\')[1]
                    except Exception:
                        material = root_pl.parts[root_pl.parts.index('Raw') + 1]
                else:
                    material = root.split(rf'/{date_token}')[0].split('Raw/')[1]
            else:
                try:
                    material = root_pl.parts[root_pl.parts.index(date_token) - 1]
                except Exception:
                    material = 'unknown'

            key = rf'{date_token}_{name}'
            data_files[key] = {
                'Date': date_token,
                'Material': material,
                'Raw_path': root
            }
            continue

        if 'Processed' in path_to_files:
            tail = os.path.basename(root).lower()
            if tail in ('inlens', 'esb'):
                continue
            has_stack_here = any('stack' in f for f in file_list)
            esb_stacks = glob.glob(os.path.join(root, 'ESB', '*stack*.tif'))
            print(f"ESB stack: {esb_stacks}")
            if not has_stack_here and len(esb_stacks) == 0:
                continue
            if found_date is None:
                continue

            date_token = found_date
            name = os.path.split(root)[1]

            if root.find(date_token) > root.find('Processed'):
                if isinstance(root_pl, pathlib.WindowsPath):
                    material = root.split(rf'\{date_token}')[0].split('Processed\\')[1]
                else:
                    material = root.split(rf'/{date_token}')[0].split('Processed/')[1]
            else:
                if isinstance(root_pl, pathlib.WindowsPath):
                    material = root.split('Reference data\\')[1].split('\\')[0]
                else:
                    material = root.split('Reference data/')[1].split('/')[0]

            key = rf'{date_token}_{name}'
            print(f"Key:{key}")
            data_files[key] = {
                'Date': date_token,
                'Material': material,
                'Processed_path': root,
                'data': {}
            }
            if load_data:
                data_files[key]['data'] = data(data_files[key]['Processed_path'], force_uint8=uint8, is_zeiss=is_zeiss)
            else:
                data_files[key]['data']['stack_meta'] = data(data_files[key]['Processed_path'], is_zeiss=is_zeiss).stack_meta
            data_files[key]['Raw_path'] = root.replace('Processed', 'Raw') if os.path.exists(root.replace('Processed', 'Raw')) else 'not known at this address'
    return data_files

def load(folder, AC=True, register=True, calib=None, uint8=False, is_zeiss=False):
    """
    Load SEM stacks from a folder (Processed or Raw) with optional alignment, calibration, and Zeiss/FEI compatibility.
    Returns: stack, stack_meta, eV, dtype_info, name, coeffs, stack_filename
    """
    slash = slash_type(folder)
    name = os.path.split(folder)[1]
    # Processed folder
    if 'Processed' in folder:
        # TIFF stack directly
        if 'tiff' in folder:
            if os.path.exists(rf'{os.path.split(folder)[0]}{slash}Metadata'):
                stack = tf.imread(folder)
                stack_file = folder
                stack_filename = os.path.split(stack_file)[1].split('.tiff')[0]
                folder = os.path.split(folder)[0]
        else:
            # Look for stack files inside Processed folder
            if os.path.exists(rf'{folder}{slash}Metadata'):
                stacks = glob.glob(rf'{folder}{slash}*stack*.tiff')
                if len(stacks) == 0:
                    stacks = glob.glob(rf'{folder}{slash}ESB{slash}*stack*.tiff')
                if len(stacks) == 1:
                    stack_file = stacks[0]
                    stack = tf.imread(stack_file)
                    stack_filename = os.path.split(stack_file)[1].split('.tiff')[0]
                else:
                    for qual in ['_seg_1', '_seg', '_corr', 'aligned', '_AC']:
                        matching = [s for s in stacks if qual in s]
                        if len(matching) == 1:
                            stack_file = matching[0]
                            stack = tf.imread(stack_file)
                            stack_filename = os.path.split(stack_file)[1].split('.tiff')[0]
                            break

        if uint8:
            stack = img_as_ubyte(stack)
        dtype_info = np.iinfo(stack.dtype)

        # Load metadata JSON
        stack_meta_files = glob.glob(rf'{folder}{slash}Metadata{slash}*stack_meta*.json')
        if len(stack_meta_files) == 0 and '_AC_' in os.path.split(folder)[1]:
            stack_meta_files = glob.glob(rf'{folder.split("_AC")[0]}{slash}Metadata{slash}*stack_meta*.json')
            with open(stack_meta_files[0]) as file:
                stack_meta = json.load(file)
            file.close()
        else:
            with open(stack_meta_files[0]) as file:
                stack_meta = json.load(file)
            file.close()

        # Determine system & analyser
        sys, analyser = sys_type(stack_meta['img1'])
        ana_voltage = []

        if is_zeiss:
            if sys == True:
                for page in stack_meta:
                    ana_voltage.append(stack_meta[page]['TLD']['Mirror'])
            coeffs = np.array(1)
            eV = np.array(np.array(ana_voltage) * coeffs)
        else:
            if sys == True:
                for page in stack_meta:
                    ana_voltage.append(stack_meta[page]['TLD']['Mirror'])
            else:
                if 'Deflector' in stack_meta['img1']['TLD']:
                    for page in stack_meta:
                        ana_voltage.append(stack_meta[page]['TLD']['Deflector'])
                else:
                    print('Warning, no Deflector in stack_meta, searching raw data for Log.csv')
                    ana_voltage = np.loadtxt(rf"{folder.replace('Processed','Raw')}{slash}Log.csv",delimiter=',', skiprows=2)[:,1]
           
            if sys == True:
                if calib is None:
                    csv_file_path = calib_file(folder)
                    if '.csv' in csv_file_path:
                        coeffs = np.loadtxt(csv_file_path)
                    if csv_file_path == 'no calibration file':
                        coeffs = np.array((-0.39866667, 6))
                if type(calib) is str:
                    coeffs = np.loadtxt(calib)
                eV = np.array(np.polyval(coeffs, ana_voltage))
            else:
                coeffs = np.array(1 / 2.84)
                eV = np.array(np.array(ana_voltage) * coeffs)
            print("Calibration Done!")
        
        return stack, stack_meta, eV, dtype_info, name, coeffs, stack_filename

    # Raw folder 
    if 'Raw' in folder:
        stack_filename = f"{name}_stack"
        inlens_files, esb_files = folder_finder(folder)

        if len(inlens_files) > 0 and len(esb_files) > 0:
            # Reference ESB image
            esb_ref = tf.imread(esb_files[-1])
            H, W = esb_ref.shape
            dtype_info = np.iinfo(esb_ref.dtype)

            # System type from first InLens file
            _, metadata = load_single_file(inlens_files[0], load_img=False)
            sys, analyser = sys_type(metadata)

            # Load InLens + ESB stacks
            inlens_stack = [img_as_float(tf.imread(f)) for f in inlens_files]
            esb_stack = [img_as_float(tf.imread(f)) for f in esb_files]

            # ROI/template selection
            if len(glob.glob(rf'{folder}{slash}*.roi')) == 1:
                roi_file = glob.glob(rf'{folder}{slash}*.roi')[0]
                roi_name = os.path.split(roi_file)[1].split('.')[0]
                temp_roi = load_roi_file(roi_file)
                yi = temp_roi[roi_name]['top']
                xi = temp_roi[roi_name]['left']
                template = inlens_stack[-1][yi:yi + temp_roi[roi_name]['height'],
                                             xi:xi + temp_roi[roi_name]['width']]
            else:
                template, temp_path, _ = template_crop(inlens_stack[-1], H, W, metadata['Scan']['HorFieldsize'])
                xi, yi = int(temp_path[0, 0]), int(temp_path[0, 1])

            # Align InLens stack
            shifts = []
            ref_inlens = inlens_stack[-1]
            for img_inlens in inlens_stack:
                if register:
                    try:
                        _, shift_y, shift_x = align_img_template(ref_inlens, img_inlens, template, H, W, yi, xi)
                    except:
                        _, shift_y, shift_x = align_img_transform(ref_inlens, img_inlens)
                else:
                    shift_x, shift_y = 0.0, 0.0
                                
                shifts.append((float(shift_x), float(shift_y)))

            # Apply shifts to ESB stack
            imgs = []
            for (shift_y, shift_x), img_esb in zip(shifts, esb_stack):
                tform = transform.EuclideanTransform(translation=(shift_x, shift_y))
                reg_esb = transform.warp(img_esb, tform, preserve_range=True)
                imgs.append(reg_esb)

            # Crop registered stack
            if register:
                shift_arr = np.array(shifts)
                x_max, y_max = np.ceil(np.max(shift_arr, axis=0))
                x_min, y_min = np.floor(np.min(shift_arr, axis=0))
                x_min, y_min = max(x_min, 0), max(y_min, 0)
                stack_f = np.array(imgs)[:, int(0 - y_min):int(H - y_max), int(0 - x_min):int(W - x_max)]
            else:
                stack_f = np.array(imgs)[:, :H, :W]

            # Build stack metadata and merge inlens metadata
            stack_meta = {}
            ana_voltage = []
            for i, (fi, fe, (shift_x, shift_y)) in enumerate(zip(inlens_files, esb_files, shifts), start=1):
                _, metadata_inlens = load_single_file(fi, load_img=False)
                _, metadata_esb = load_single_file(fe, load_img=True)
                page = f"img{i}"
                stack_meta[page] = metadata_esb.copy()
                stack_meta[page]['InLens'] = metadata_inlens
                stack_meta[page]['Processing'] = {
                    'file_inlens': fi,
                    'file_esb': fe,
                    'transformation': {'x': float(shift_x), 'y': float(shift_y)},
                    'temp_match': {'xi': float(xi), 'yi': float(yi)}
                }
                ana_voltage.append(metadata_esb['TLD'][analyser])

            # Convert to original dtype
            stack = np.array(stack_f * dtype_info.max, dtype=dtype_info.dtype)

            # Energy calibration
            if is_zeiss:
                if sys == True:
                    coeffs = np.array(1)
                    eV = np.array(np.array(ana_voltage) * coeffs)
                else:
                    print('Warning, Zeiss data should be ordered in V, setting sys=True for eV conversion')
            else:
                if sys == True:
                    if calib is None:
                        csv_file_path = calib_file(folder)
                        if '.csv' in csv_file_path:
                            coeffs = np.loadtxt(csv_file_path)
                        if csv_file_path == 'no calibration file':
                            coeffs = np.array((-0.39866667, 6))
                    elif isinstance(calib, str):
                        coeffs = np.loadtxt(calib)
                    eV = np.array(np.polyval(coeffs, ana_voltage))
                else:
                    coeffs = np.array(1 / 2.84)
                    eV = np.array(np.array(ana_voltage) * coeffs)

            return stack, stack_meta, eV, dtype_info, name, coeffs, stack_filename

def reformat_zeiss_metadata(zeiss_metadata):
    """
    Add missing metadata fields to Zeiss metadata to match FEI/ThermoFisher structure.
    """
    zeiss_metadata['System'] = {}
    zeiss_metadata['System']['SystemType'] = 'Zeiss'
    store_resolution = zeiss_metadata['dp_image_store'][1].split(' * ')
    zeiss_metadata['Image'] = {}
    zeiss_metadata['Image']['ResolutionX'] = int(store_resolution[0]) # in pixels
    zeiss_metadata['Image']['ResolutionY'] = int(store_resolution[1]) # in pixels
    zeiss_metadata['Image']['Average'] = zeiss_metadata['ap_frame_average_count'][1]  # in frames - likely 1
    zeiss_metadata['Image']['Integrate'] = zeiss_metadata['ap_frame_int_count'][1]  # in frames - likely 0
    zeiss_metadata['EBeam'] = {}
    zeiss_metadata['EBeam']['BeamCurrent'] = zeiss_metadata['ap_beam_current'][1] / 1e6  # in uA -> A   #OR ap_manualcurrent OR ap_beam_current_monitor
    zeiss_metadata['EBeam']['BeamMode'] = 'Standard'  # TODO not available in Zeiss metadata - define default or get from app?
    zeiss_metadata['Beam'] = {}
    zeiss_metadata['Beam']['HV'] = zeiss_metadata['ap_actualkv'][1] * 1e3 # in kV -> V  # EHT target is HV?
    zeiss_metadata['Scan'] = {}
    zeiss_metadata['Scan']['HorFieldsize'] = zeiss_metadata['ap_width'][1] / 1e6 # in µm to m   # effectively store res * pixel size
    zeiss_metadata['Scan']['VerFieldsize'] = zeiss_metadata['ap_height'][1] / 1e6 # in µm to m   # effectively store res * pixel size
    zeiss_metadata['Scan']['Dwelltime'] = zeiss_metadata['dp_dwell_time'][1] / 1e9 # in ns to s   # NOTE different name to EScan
    zeiss_metadata['Scan']['PixelWidth'] = zeiss_metadata['ap_pixel_size'][1] / 1e9 # in nm to m
    zeiss_metadata['Scan']['PixelHeight'] = zeiss_metadata['ap_pixel_size'][1] / 1e9 # in nm to m
    zeiss_metadata['EScan'] = {}
    zeiss_metadata['EScan']['HorFieldsize'] = zeiss_metadata['ap_width'][1] / 1e6 # in µm to m   # effectively store res * pixel size
    zeiss_metadata['EScan']['VerFieldsize'] = zeiss_metadata['ap_height'][1] / 1e6 # in µm to m   # effectively store res * pixel size
    zeiss_metadata['EScan']['Dwell'] = zeiss_metadata['dp_dwell_time'][1] / 1e9 # in ns to s
    zeiss_metadata['EScan']['PixelWidth'] = zeiss_metadata['ap_pixel_size'][1] / 1e9 # in nm to m
    zeiss_metadata['EScan']['PixelHeight'] = zeiss_metadata['ap_pixel_size'][1] / 1e9 # in nm to m
    zeiss_metadata['EScan']['ScanInterlacing'] = 16     # TODO not available in Zeiss metadata - define default or get from app?
    zeiss_metadata['Stage'] = {}
    zeiss_metadata['Stage']['StageR'] = zeiss_metadata['ap_stage_at_r'][1] * (np.pi/180) # in degrees to radians    # NOTE convention may be different - CONFIRM
    zeiss_metadata['Stage']['StageT'] = zeiss_metadata['ap_stage_at_t'][1] * (np.pi/180) # in degrees to radians    # NOTE convention may be different - CONFIRM
    zeiss_metadata['Stage']['StageX'] = zeiss_metadata['ap_stage_at_x'][1] / 1e3 # in mm to m
    zeiss_metadata['Stage']['StageY'] = zeiss_metadata['ap_stage_at_y'][1] / 1e3 # in mm to m
    zeiss_metadata['Stage']['StageZ'] = zeiss_metadata['ap_stage_at_z'][1] / 1e3 # in mm to m
    zeiss_metadata['Stage']['WorkingDistance'] = zeiss_metadata['ap_wd'][1] / 1e3 # in mm to m
    zeiss_metadata['TLD'] = {}
    zeiss_metadata['TLD']['Mirror'] = zeiss_metadata['ap_esb_grid'][1] # in Volts
    zeiss_metadata['TLD']['Deflector'] = zeiss_metadata['ap_esb_grid'][1] # in Volts
    zeiss_metadata['TLD']['Brightness'] = zeiss_metadata['ap_brightness'][1] # in %   #- There is A,B,C and D brightness?
    zeiss_metadata['TLD']['Contrast'] = zeiss_metadata['ap_contrast'][1] # in %  #- There is A,B,C and D brightness?
    zeiss_metadata['Vacuum'] = {}
    zeiss_metadata['Vacuum']['ChPressure'] = zeiss_metadata['ap_system_vac'][1] # in mbar - could also be gun vacuum but unlikely here?
    return zeiss_metadata

def load_single_file(file, load_img=True, crop_footer=False):
    with tf.TiffFile(file) as tif:
        try:
            metadata = tif.fei_metadata
            system = metadata['System']['SystemType']
        except:
            # Fallback, try Zeiss format as the metadata format failed for FEI/Thermo
            metadata = tif.sem_metadata
            system = metadata['dp_sem'][1] # Zeiss system type location
            metadata = reformat_zeiss_metadata(metadata)
    tif.close()

    if load_img == True:
        img = tf.imread(file)
        if crop_footer == True:
            yc = metadata['Image']['ResolutionY']
            img = img[:yc,:]
            print("Single file loading")
        return img, metadata
    else:
        return None, metadata
    
def slash_type(path):
    if type(pathlib.Path(path)) is pathlib.WindowsPath:
        slash = '\\'
    else:
        slash = '/'
    return slash

def sys_type(metadata):
    """
    Gives system type of the FEI/ThermoFisher or Zeiss microscope
    and analyser type 'Deflector' for Nova, 'Mirror' for Helios, 'Mirror'/'Deflector' for Zeiss
    """
    if 'Helios' in metadata['System']['SystemType']:
        sys = True
        analyser = 'Mirror'
    if 'Nova' in metadata['System']['SystemType']:
        sys = False
        analyser = 'Deflector'
    if 'Zeiss' in metadata['System']['SystemType']:
        sys = True  # Treat Zeiss as the V are ordered in filenames
        analyser = 'Deflector' 
    return sys, analyser

def template_crop(ref_img, y, x, hfw):
    w = hfw * 1e6
    area = w * 3 / 1040 + 303 / 520
    height, width = int(area * y), int(area * x)
    yi, yii = int(y / 2 - height / 2), int(y / 2 + height / 2)
    xi, xii = int(x / 2 - width / 2), int(x / 2 + width / 2)
    temp_path = np.array([[xi, yi], [xii, yi], [xii, yii], [xi, yii], [xi, yi]])
    return ref_img[yi:yii, xi:xii], temp_path, area

"""
Image alignment.
PCC, Translation, Template Matching.
"""
# Translation
def align_img_transform(ref_img, mov_img):
    im1 = ref_img
    im2 = mov_img
    im1 = (im1 - np.min(im1)) / (np.ptp(im1) + 1e-9)
    im2 = (im2 - np.min(im2)) / (np.ptp(im2) + 1e-9)
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    h, w = ref_img.shape
    number_of_iterations = 5000
    termination_eps = 1e-10
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    cc, warp_matrix = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria)
    shift_x = warp_matrix[0, 2]
    shift_y = warp_matrix[1, 2]
    reg_img = cv2.warpAffine(mov_img, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return reg_img, shift_y, shift_x

# Template Matching
def align_img_template(ref_img, mov_img, template, y, x, yi, xi):
    if y is None:
        y = ref_img.shape[0]
    if x is None:
        x = ref_img.shape[1]
    mov_img = mov_img[0:y, 0:x]
    result = match_template(np.array(mov_img, dtype='uint8'), np.array(template, dtype='uint8'), TM_CCOEFF_NORMED)
    minV, maxV, minpt, maxpt = min_max_loc(result)
    shift_x, shift_y = np.asarray(maxpt) - (xi, yi)
    tform = transform.EuclideanTransform(translation=(shift_x, shift_y))
    reg_img = transform.warp(mov_img, tform)
    # reg_img = np.array(reg_img)
    return reg_img, shift_y, shift_x
 
# PCC - phase cross correlation
def align_img_pcc(ref_img, mov_img, crop_y=None, crop_x=None, upsample_factor=10):
    if crop_y is None:
        crop_y = ref_img.shape[0]
    if crop_x is None:
        crop_x = ref_img.shape[1]
    mov_img = mov_img[0:crop_y, 0:crop_x]
    shift, err, phasediff = pcc(ref_img[0:crop_y, 0:crop_x], mov_img, upsample_factor=upsample_factor)
    shift_y, shift_x = shift
    tform = transform.EuclideanTransform(translation=(shift_x, shift_y))
    reg_img = transform.warp(mov_img, tform)
    reg_img = np.array(reg_img)
    return reg_img, shift_y, shift_x

# EsB grid voltages from stack metadata
def MV(stack_meta):
    MV= []
    for page in stack_meta:
        MV.append(stack_meta[page]['TLD']['Mirror'])
    MV = np.array(MV)
    return MV

# Calibration File
def calib_file(path, filename=None):
    slash = slash_type(path)
    if filename is None:
        filename = 'calibration.csv'
    while not os.path.exists(rf'{os.path.split(path)[0]}{slash}{filename}'):
        path = os.path.split(path)[0]
        if len(path) < 6:
            abort=True
            return 'no calibration file'
            break
    # if abort is True:
    #     return 'no calibration file'
    return rf'{os.path.split(path)[0]}{slash}{filename}'

def conversion(stack_meta, factor, corr):
    eV = (np.array(MV(stack_meta)) * factor) + corr
    print("Calibrating")
    return eV

#  Generates axes labels for SE/LL-BSE spectrum plot
def plot_axes(norm=False, x_eV=True):
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'sans:italic:bold'
    plt.rcParams['mathtext.bf'] = 'sans:bold'
    if x_eV:
        plt.xlabel('Energy, $\mathit{E}$ [eV]', weight='bold')
    else:
        plt.xlabel('ESB grid voltage [V]', weight='bold')
        plt.gca().invert_xaxis()
    if not norm:
        plt.ylabel('Emission intensity [arb.u.]', weight='bold')
    else:
        plt.ylabel('Emission intensity norm. [arb.u.]', weight='bold')

def plot_scalebar(img, length_fraction=0.3, font_size=15, stack_meta=None, metadata=None, pixel_width=None, hfw=None, img_info=None, save_path=None, plot=True):
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    if stack_meta is not None:
        pixel_width = stack_meta['img1']['Scan']['PixelWidth']
    if metadata is not None:
        pixel_width = metadata['Scan']['PixelWidth']
    if hfw is not None:
        pixel_width = hfw / img.shape[1]
    if img_info is not None:
        with tf.TiffFile(img_info) as tif:
            pixel_width = tif.fei_metadata['Scan']['PixelWidth']
        tif.close()
    # else:
    #     print('Provide pixel width info')
    #     return    
    scalebar = ScaleBar(dx=pixel_width, length_fraction=length_fraction, location='lower right', border_pad=0.5, color='w', box_color='k', box_alpha=0.35, font_properties={'size': str(font_size)})
    plt.gca().add_artist(scalebar)
    if save_path is not None:
        plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0)
    if plot is True:
        plt.show()

# def zpro(stack):
#     z = []
#     for i in np.arange(0, stack.shape[0], 1):
#         z.append(np.mean(stack[i, :, :]))
#     return z

def zpro(stack):
    return [float(np.mean(stack[:,:,i])) for i in range(stack.shape[2])]

def spec_dose(stack_meta):
    if 'User' in stack_meta:
        img_meta = stack_meta
        stack_meta = {}
        stack_meta['img1'] = img_meta
    d_img_list = []
    for page in stack_meta:
        I_0 = stack_meta[page]['EBeam']['BeamCurrent']
        t_dwell = stack_meta[page]['Scan']['Dwelltime']
        n_px = stack_meta[page]['Image']['ResolutionX'] * stack_meta[page]['Image']['ResolutionY']
        n_average = stack_meta[page]['Image']['Average']
        n_integrate = stack_meta[page]['Image']['Integrate']
        A = stack_meta[page]['Scan']['HorFieldsize'] * stack_meta[page]['Scan']['VerFieldsize']
        d_img = ((I_0 * t_dwell * n_px * (n_average + n_integrate)) / A)
        d_img_list.append(d_img)
    d_spec = np.sum(d_img_list)
    if 'Processing' in stack_meta[page]:
        if 'angular_correction' in stack_meta[page]['Processing']:
            d_spec = 2 * d_spec
    return d_spec

def norm(data_arr, n_min=False):
    if n_min is False:
        data_n = data_arr / np.max(data_arr)
    else:
        data_n = (data_arr - np.min(data_arr)) / (np.max(data_arr) - np.min(data_arr))
    return data_n

def roi_masks(img, rois_data):
    if type(img) is str:
        img = tf.imread(img)
    if len(img.shape) == 3:
        img_r = img[-1,:,:]
        z,y,x = img.shape
    if len(img.shape) == 2:
        img_r = img
        y,x = img.shape
    
    if '.zip' in rois_data or '.roi' in rois_data:
        rois = load_roi_file(rois_data)
        for name in rois:
            ygrid, xgrid = np.mgrid[:y, :x]
            xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
            img_mask = np.ma.getmask(np.ma.array(img_r, mask=False))    # initialise the False 2D mask that roi_paths will be added to
            for key in rois[name]['roi_path']:                          # loop through rois in composite roi
                pth = Path(rois[name]['roi_path'][key], closed=False)       # construct a Path from the vertices
                mask = pth.contains_points(xypix)                           # test which pixels fall within the path
                mask = mask.reshape(y,x)                                    # reshape to the same size as the image
                img_mask = np.ma.mask_or(img_mask,mask)                     # add the xycrop to the 2D mask
            rois[name]['img_mask'] = img_mask                           # add img mask to rois dict
    elif 'type' in rois_data:
        rois={}
        if 'img_mask' in rois_data:
            rois[0] = rois_data
        else:
            ygrid, xgrid = np.mgrid[:y,:x]
            xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
            img_mask = np.ma.getmask(np.ma.array(img_r, mask=False))
            pth = Path(rois_data['roi_path'], closed=False)
            mask = pth.contains_points(xypix)
            mask = mask.reshape(y,x)
            img_mask = np.ma.mask_or(img_mask,mask)
            rois[0]={}
            rois[0]['img_mask'] = img_mask
    elif isinstance(rois_data, collections.abc.Mapping) is False:
        rois=rois_data
        for name in rois:
            ygrid, xgrid = np.mgrid[:y, :x]
            xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
            img_mask = np.ma.getmask(np.ma.array(img_r, mask=False))
            pth = Path(rois[name]['roi_path'], closed=False)
            mask = pth.contains_points(xypix)
            mask = mask.reshape(y,x)
            img_mask = np.ma.mask_or(img_mask,mask)
            rois[name]['img_mask'] = img_mask
    elif isinstance(rois_data, collections.abc.Mapping) is True:
        rois = rois_data
        for name in rois:
            if 'img_mask' in rois[name]:
                continue
            else:
                ygrid, xgrid = np.mgrid[:y, :x]
                xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
                img_mask = np.ma.getmask(np.ma.array(img_r, mask=False))
                for key in rois[name]['roi_path']:
                    pth = Path(rois[name]['roi_path'][key], closed=False)
                    mask = pth.contains_points(xypix)
                    mask = mask.reshape(y,x)
                    img_mask = np.ma.mask_or(img_mask,mask)
                rois[name]['img_mask'] = img_mask
    elif '.npy' in rois_data:
        rois={}
        masks = np.load(rois_data)
        i=0
        while i <= masks.max():
            rois[i] = {}
            rois[i]['img_mask'] = np.where(masks==i,True,False)
            i+=1
    elif type(rois_data) is np.ndarray:
        rois={}
        ygrid, xgrid = np.mgrid[:y,:x]
        xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T
        img_mask = np.ma.getmask(np.ma.array(img_r, mask=False))
        pth = Path(rois_data, closed=False)
        mask = pth.contains_points(xypix)
        mask = mask.reshape(y,x)
        img_mask = np.ma.mask_or(img_mask,mask)
        rois[0]={}
        rois[0]['img_mask'] = img_mask
    else:
        rois=rois_data
    return rois
#  Gets the xy points to draw a roi from the imageJ .roi file to a python dictionary the read_roi module can not write imageJ compatible roi files.
def load_roi_file(path_to_roi_file):
    if '.zip' in path_to_roi_file:
        r=read_roi.read_roi_zip(path_to_roi_file)
    if '.roi' in path_to_roi_file:
        r=read_roi.read_roi_file(path_to_roi_file)
    for name in r:
        xy_crop = {}
        if r[name]['type'] == 'rectangle':
            x1 = r[name]['left']
            x2 = x1 + r[name]['width']
            y1 = r[name]['top']
            y2 = y1 + r[name]['height']
            xc = np.array([x1,x2,x2,x1])
            yc = np.array([y1,y1,y2,y2])
            xy_crop[0] = np.vstack((xc, yc)).T
        if r[name]['type'] == 'composite':
            for i,path in enumerate(r[name]['paths']):
                xc = np.array(r[name]['paths'][i])[:,0]
                yc = np.array(r[name]['paths'][i])[:,1]
                xy_crop[i] = np.vstack((xc, yc)).T
        if r[name]['type'] == 'freehand':
            xc = np.array(r[name]['x'])
            yc = np.array(r[name]['y'])
            xy_crop[0] = np.vstack((xc, yc)).T
        if r[name]['type'] == 'oval':
            rad = r[name]['width']/2
            centre = np.array([r[name]['left']+rad, r[name]['top']+rad])
        r[name]['roi_path'] = xy_crop
    return r

class data:
    def __init__(self, folder, AC=True, calib=None, reg=True, force_uint8=False, is_zeiss=False):
        slash = slash_type(folder)
        date = None
        for part in pathlib.Path(folder).parts:
            m = regex.search(r"(\d{6}[\w-]*)", part)
            if m is not None:
                date = m.group(0)
                break
        self.date = date
        if AC is True and 'Raw' in folder and os.path.exists(rf'{folder}_R'):
            stack, stack_meta, self.eV, self.dtype_info, name, coeffs, stack_filename = load(folder, calib=calib, register=reg, uint8=force_uint8, is_zeiss=is_zeiss)
            stack_r_file = rf'{folder}_R'
            stack_r, stack_meta_r, eV_r, dtype_info_r, name_r, coeffs_r, stack_filename_r = load(stack_r_file, calib=calib, register=reg, uint8=force_uint8, is_zeiss=is_zeiss)
            self.folder = rf'{folder}_AC'
            for page, page_r in zip(stack_meta, stack_meta_r):
                stack_meta[page]['Processing']['angular_correction'] = 'True'
                stack_meta[page]['Processing']['transformation_r'] = stack_meta_r[page]['Processing']['transformation']
                file_r = stack_meta_r[page]['Processing'].get('file_esb', stack_meta_r[page]['Processing'].get('file'))
                stack_meta[page]['Processing']['file_r'] = file_r
            stack = img_as_float(stack)
            stack_r = img_as_float(stack_r)
            temp1 = stack[-1, :, :]
            temp2 = stack_r[-1, :, :]
            temp2 = transform.rotate(temp2, 180)
            stack_r = scnd.rotate(stack_r, 180, axes=(2, 1))
            y1, x1 = temp1.shape
            y2, x2 = temp2.shape
            if y1 == y2:
                yicrop = 'Equal'
                py = 0
                yi = temp1.shape[0]
            if y1 < y2:
                py = y2 - y1
                temp1 = np.insert(temp1, (y2 - y1) * [0], 0, axis=0)
                stack = np.insert(stack, (y2 - y1) * [0], 0, axis=1)
                yi = temp1.shape[0]
                yicrop = True
            if y1 > y2:
                py = y1 - y2
                temp2 = np.insert(temp2, (y1 - y2) * [0], 0, axis=0)
                stack_r = np.insert(stack_r, (y1 - y2) * [0], 0, axis=1)
                yi = temp2.shape[0]
                yicrop = False
            if x1 == x2:
                xicrop = 'Equal'
                px = 0
                xi = temp1.shape[1]
            if x1 < x2:
                px = x2 - x1
                temp1 = np.insert(temp1, (x2 - x1) * [0], 0, axis=1)
                stack = np.insert(stack, (x2 - x1) * [0], 0, axis=2)
                xi = temp1.shape[1]
                xicrop = True
            if x1 > x2:
                px = x1 - x2
                temp2 = np.insert(temp2, (x1 - x2) * [0], 0, axis=1)
                stack_r = np.insert(stack_r, (x1 - x2) * [0], 0, axis=2)
                xi = temp2.shape[1]
                xicrop = False

            shifts, err, phasediff = pcc(temp1, temp2)
            ty, tx = shifts
            stack_r = scnd.shift(stack_r, shift=[0, ty, tx])
            halfDiff = (stack - stack_r) / 2
            stackCorr = stack - halfDiff
            stackCorr = stackCorr * self.dtype_info.max
            stack_AC = stackCorr.astype(self.dtype_info.dtype)
            if yicrop == 'Equal':
                if ty > 0:
                    stack_AC_crop = stack_AC[:, int(py + ty):int(yi - ty), :]
                if ty <= 0:
                    y0 = 0
                    if abs(ty) < py:
                        y0 = int(py)
                    stack_AC_crop = stack_AC[:, y0:int(yi + ty), :]
            if yicrop is False:
                if ty > 0:
                    stack_AC_crop = stack_AC[:, int(py + ty):int(yi + ty), :]
                if ty <= 0:
                    y0 = 0
                    if abs(ty) < py:
                        y0 = int(py)
                    stack_AC_crop = stack_AC[:, y0:int(yi + ty), :]
            if yicrop is True:
                if ty > py:
                    y0 = int(ty)
                else:
                    y0 = py
                stack_AC_crop = stack_AC[:, y0:int(yi + ty), :]
            if xicrop == 'Equal':
                if tx > 0:
                    stack_AC_crop = stack_AC_crop[:, :, :]
                if tx <= 0:
                    x0 = 0
                    if abs(tx) < px:
                        x0 = int(px)
                    stack_AC_crop = stack_AC_crop[:, :, x0:int(xi + tx)]
            if xicrop is False:
                if tx > 0:
                    stack_AC_crop = stack_AC_crop[:, :, int(tx + px):xi]
                if tx <= 0:
                    x0 = 0
                    if abs(tx) < px:
                        x0 = int(px)
                    stack_AC_crop = stack_AC_crop[:, :, x0:int(xi + tx)]
            if xicrop is True:
                if tx > px:
                    x0 = int(tx)
                if tx > 0:
                    x0 = 0
                    stack_AC_crop = stack_AC_crop[:, :, x0:xi]
                if tx <= 0:
                    x0 = px
                    stack_AC_crop = stack_AC_crop[:, :, x0:int(xi + tx)]
            self.stack = stack_AC_crop
            self.stack_meta = stack_meta
            self.stack_meta_r = stack_meta_r
            self.name = rf'{name}_AC'
            self.shape = self.stack.shape
            self.coeffs = coeffs
            self.stack_filename = rf'{stack_filename}_AC'
        else:
            self.folder = folder
            self.stack, self.stack_meta, self.eV, self.dtype_info, self.name, self.coeffs, self.stack_filename = load(folder, calib=calib, AC=False, register=reg, uint8=force_uint8, is_zeiss=is_zeiss)
            self.shape = self.stack.shape

#  The indexes of values within eV ranges defined by xmin and xmax.
    def rows(self, xlim=[0, 800], x_eV=True): # Change xlim=[]
        if x_eV:
            x = self.eV
        else:
            x = MV(self.stack_meta)
        if type(xlim) is list:
            rows = np.where((x >= xlim[0]) & (x <= xlim[1]))[0]
        if xlim == 'all':
            rows = self.eV.argsort()
        return rows
    
    def mv(self):
        return MV(self.stack_meta)
    
    def spec(self, rois=None, pixel_spec:int = None):
        if rois is not None:
            if pixel_spec is None:
                r = roi_masks(self.stack, rois)
            else:
                r = roi_masks(smooth.uniform(self.stack, size=pixel_spec), rois)
            for name in r:
                stack_mask = np.array([r[name]['img_mask']] * self.shape[0])
                if pixel_spec is None:
                    stack_masked = np.ma.masked_array(self.stack, ~stack_mask)
                    r[name]['spec'] = np.gradient(zpro(stack_masked))
                else:
                    stack_masked = np.ma.masked_array(smooth.uniform(self.stack, size=pixel_spec), ~stack_mask)
                    r[name]['spec_pix'] = pd.DataFrame()
                    x_min = int(np.min(r[name]['roi_path'][0][:, 0]))
                    x_max = int(np.max(r[name]['roi_path'][0][:, 0]))
                    y_min = int(np.min(r[name]['roi_path'][0][:, 1]))
                    y_max = int(np.max(r[name]['roi_path'][0][:, 1]))
                    for yi in range(y_min, y_max, 1):
                        for xi in range(x_min, x_max, 1):
                            zpro_pix = stack_masked[:, yi, xi]
                            spec_pix = np.gradient(zpro_pix)
                            r[name]['spec_pix'][f'x{xi},y{yi}'] = spec_pix
                    r[name]['spec_avg'] = r[name]['spec_pix'].mean(axis=1)
                    r[name]['spec_sd'] = r[name]['spec_pix'].std(axis=1)
                    r[name]['pixel_spec_width'] = pixel_spec
            return r
        else:
            return np.gradient(zpro(self.stack))
        
    def spec_dose(self):
        return spec_dose(self.stack_meta)
    
    def plot_template_roi(self):
        ref_img = self.stack_meta[f'img{len(self.stack)}']['Processing']['temp_match']['ref_img']
        temp_path = self.stack_meta[f'img{len(self.stack)}']['Processing']['temp_match']['path']
        area = self.stack_meta[f'img{len(self.stack)}']['Processing']['temp_match']['area']
        plt.imshow(ref_img)
        plt.plot(temp_path[:, 0], temp_path[:, 1], c='r')
        plt.text(temp_path[0, 0] + 10, temp_path[0, 1] - 15, rf'template, area = {round(area, 3)}', backgroundcolor=(1, 1, 1, 0.3))
        plt.show()

    def reg_tforms(self):
        shifts = {}
        shift_x = []
        shift_y = []
        for page in self.stack_meta:
            shift_x.append(self.stack_meta[page]['Processing']['transformation']['x'])
            shift_y.append(self.stack_meta[page]['Processing']['transformation']['y'])
        shifts[0] = np.array([shift_x, shift_y]).T
        if '_AC' in self.name and 'angular_correction' in self.stack_meta['img1']['Processing']:
            shift_x_r = []
            shift_y_r = []
            for page in self.stack_meta:
                shift_x_r.append(self.stack_meta[page]['Processing']['transformation_r']['x'])
                shift_y_r.append(self.stack_meta[page]['Processing']['transformation_r']['y'])
            shifts[1] = np.array([shift_x_r, shift_y_r]).T
        return shifts
    
    def img_avg(self):
        img = np.array(np.mean(self.stack, axis=0), dtype=self.dtype_info.dtype)
        return img
    
    def plot_img(self, scalebar=True, plot=True, fin_img=False):
        if not fin_img:
            img = data.img_avg(self)
        else:
            img = self.stack[-1, :, :]
        if scalebar:
            plot_scalebar(img, stack_meta=self.stack_meta, plot=plot)
        else:
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            if plot:
                plt.show()

    def plot_spec(self, rois=None, groups: dict or str = None, plot=True, x_eV=True, xlim=[-1, 8], pixel_spec: int = None, smooth_width: int = None, savefig=False):
        slash = slash_type(self.folder)
        xlim = np.array(xlim)
        if x_eV:
            rows = np.where((self.eV >= xlim[0]) & (self.eV <= xlim[1]))[0]
        else:
            rows = np.where((MV(self.stack_meta) >= xlim[0]) & (MV(self.stack_meta) <= xlim[1]))[0]
        if rois is None:
            if smooth_width is None:
                y = data.spec(self)
            else:
                y = smooth.mov_av(data.spec(self), smooth_width)
            if x_eV:
                plt.plot(self.eV[rows], y[rows])
            else:
                plt.plot(MV(self.stack_meta)[rows], y[rows])
            plot_axes(x_eV=x_eV)
            if isinstance(savefig, str):
                save_path = savefig
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(rf'{save_path}{slash}spec.png', dpi=300, transparent=True)
            if savefig is True:
                save_path = rf'{self.folder}{slash}ROI'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(rf'{save_path}{slash}spec.png', dpi=300, transparent=True)
            if plot:
                plt.show()
            else:
                plt.close()
        if rois is not None:
            if pixel_spec is None:
                r = roi_masks(self.stack, rois)
            else:
                r = self.spec(rois, pixel_spec)
            color = cm.rainbow(np.linspace(0, 1, len(r)))
            masks = np.zeros(self.shape[1:3])
            make_group = False
            if groups is not None:
                if type(groups) is str and '.json' in groups:
                    with open(groups) as file:
                        groups = json.load(file)
                    make_group = False
                else:
                    make_group = True
            if type(groups) is dict:
                groups_df = {}
                for k in groups:
                    groups_df[k] = pd.DataFrame()
            for i, (name_r, c) in enumerate(zip(r, color)):
                if 'spec' not in r[name_r]:
                    r[name_r]['spec'] = data.spec(self, r[name_r])[0]['spec']
                    if type(groups) is dict:
                        for k in groups:
                            if name_r in groups[k]:
                                groups_df[k][name_r] = r[name_r]['spec']
                if smooth_width is None:
                    yv = r[name_r]['spec']
                else:
                    yv = smooth.mov_av(r[name_r]['spec'], smooth_width)
                if x_eV and groups is None and pixel_spec is None:
                    plt.plot(self.eV[rows], yv[rows], c=c, label=name_r)
                if not x_eV and groups is None and pixel_spec is None:
                    plt.plot(MV(self.stack_meta)[rows], yv[rows], c=c, label=name_r)
                img_mask = np.where(r[name_r]['img_mask'] == True, i + 1, 0)
                masks = masks + img_mask
            if groups is not None:
                for k in groups_df:
                    groups_df[k]['mean'] = groups_df[k].mean(axis=1)
                    groups_df[k]['std'] = groups_df[k].iloc[:, :-1].std(axis=1)
                    if smooth_width is None:
                        yv = groups_df[k]['mean']
                        std = groups_df[k]['std']
                    else:
                        yv = smooth.mov_av(groups_df[k]['mean'], smooth_width)
                        std = smooth.mov_av(groups_df[k]['std'], smooth_width)
                    if x_eV:
                        plt.plot(self.eV[rows], yv[rows], label=k)
                        plt.fill_between(self.eV[rows], yv[rows] - std[rows], yv[rows] + std[rows], alpha=0.5)
                    else:
                        mvx = MV(self.stack_meta)
                        plt.plot(mvx[rows], yv[rows], label=k)
                        plt.fill_between(mvx[rows], yv[rows] - std[rows], yv[rows] + std[rows], alpha=0.5)
            if pixel_spec is not None:
                clr = cm.rainbow(np.linspace(0, 1, len(r)))
                for j, name_r in enumerate(r):
                    if smooth_width is None:
                        yv = r[name_r]['spec_avg']
                        sd = r[name_r]['spec_sd']
                    else:
                        yv = smooth.mov_av(r[name_r]['spec_avg'], smooth_width)
                        sd = smooth.mov_av(r[name_r]['spec_sd'], smooth_width)
                    plt.plot(self.eV[rows], yv[rows], c=clr[j], label=name_r)
                    plt.fill_between(self.eV[rows], yv[rows] + sd[rows], yv[rows] - sd[rows], color=clr[j], alpha=0.5)
            masks = masks - 1
            normc = colors.Normalize(vmin=0, vmax=len(r))
            cmap = plt.get_cmap('rainbow')
            masks_n = normc(masks)
            rgba = cmap(masks_n)
            rgba[masks == -1, :] = [1, 1, 1, 0]
            plot_axes(x_eV=x_eV)
            plt.legend()
            if isinstance(savefig, str):
                save_path = savefig
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(rf'{save_path}{slash}spec.png', dpi=300, transparent=True)
                if make_group:
                    with open(rf'{save_path}{slash}groups.json', 'w') as f:
                        json.dump(groups, f)
            if savefig is True:
                save_path = rf'{self.folder}{slash}ROI'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                plt.savefig(rf'{save_path}{slash}spec.png', dpi=300, transparent=True)
                if make_group:
                    with open(rf'{save_path}{slash}groups.json', 'w') as f:
                        json.dump(groups, f)
            if plot:
                plt.show()
                self.plot_img(plot=False)
                plt.imshow(rgba, alpha=0.35)
                plt.axis('off')
                for name_r in r:
                    if len(r[name_r]['roi_path']) == 1:
                        plt.annotate(r[name_r]['name'], xy=r[name_r]['roi_path'][0][0], va='top', ha='left', c=[1, 1, 1], fontsize=8)
                    else:
                        for i in r[name_r]['roi_path']:
                            plt.annotate(r[name_r]['name'], xy=r[name_r]['roi_path'][0][0], va='top', ha='left', c=[1, 1, 1], fontsize=8)
                if isinstance(savefig, str):
                    save_path = savefig
                    plt.savefig(rf'{save_path}{slash}masks.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
                if savefig is True:
                    plt.savefig(rf'{save_path}{slash}masks.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
    
    def plot_zpro(self, x_eV=True):
        plt.plot(self.eV, zpro(self.stack))
        plot_axes(x_eV=x_eV)
        plt.show()
    
    def plot_stack_meta(self, reg, save_path=None):
        sys, analyser = sys_type(self.stack_meta['img1'])
        ChPressure = []
        V = []
        slices = []
        for i, page in enumerate(self.stack_meta):
            for i, page in enumerate(self.stack_meta):
                V.append(self.stack_meta[page]['TLD'][analyser])
                ChPressure.append(self.stack_meta[page]['Vacuum']['ChPressure'])
                slices.append(i+1)
        
        fig, axs = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios': [3, 2, 1]})
        fig.suptitle(f"{self.date}_{self.name}")

        if reg is True:
            shifts = self.reg_tforms()
            axs[0].plot(slices, shifts[0][:, 0], 'o', color='r')
            axs[0].plot(slices, shifts[0][:, 1], 'o', color='b')
            axs[0].legend(['x_shift', 'y_shift'])
            if '_AC' in self.name and 'angular_correction' in self.stack_meta['img1']['Processing']:
                axs[0].plot(slices, shifts[1][:, 0], 'ks', markerfacecolor='none', color='c')
                axs[0].plot(slices, shifts[1][:, 1], 'ks', markerfacecolor='none', color='m')
                axs[0].legend(['x_shift', 'y_shift', 'x_shift_r', 'y_shift_r'])
        
        axs[1].plot(slices, zpro(self.stack), 'k', label='Zpro.')
        axs[1].set(ylabel='slice mean')
        axs[1].legend()
        axs[2].plot(slices, ChPressure, 'k-', label='ChPres.')
        axs[2].set(xlabel='slice number', ylabel='Pa')
        axs[2].legend()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()
        if save_path is None:
            plt.show()

    def save_data(self, reg, save_path=None):
        slash = slash_type(self.folder)
        if save_path is None:
            if 'Raw' in self.folder:
                save_path = self.folder.replace('Raw', 'Processed')
            if 'Processed' in self.folder:
                save_path = self.folder
        os.makedirs(save_path, exist_ok=True)
        esb_path = rf"{save_path}{slash}ESB"
        meta_path = rf"{save_path}{slash}Metadata"
        os.makedirs(esb_path, exist_ok=True)
        os.makedirs(meta_path, exist_ok=True)
        pixel_width_um = self.stack_meta['img1']['Scan']['PixelWidth'] * 1e6
        esb_avg_img = data.img_avg(self)
        tf.imwrite(rf"{esb_path}{slash}_avg_img.tif", data=esb_avg_img, dtype=self.dtype_info.dtype, photometric='minisblack', imagej=True, resolution=(1. / pixel_width_um, 1. / pixel_width_um), metadata={'unit': 'um', 'axes': 'YX'})
        plot_scalebar(esb_avg_img, stack_meta=self.stack_meta, save_path=rf"{esb_path}{slash}_avg_img_scaled.png")
        labels = []
        for i, page in enumerate(self.stack_meta):
            if 'Helios' in self.stack_meta['img1']['System']['SystemType']:
                mv_val = self.stack_meta[page]['TLD']['Mirror']
            elif 'Nova' in self.stack_meta['img1']['System']['SystemType']:
                def_val = self.stack_meta[page]['TLD']['Deflector']
                eV_val = def_val * (1 / 2.84)
                csv_file_path = calib_file(self.folder)
                if csv_file_path == 'no calibration file':
                    coeffs = np.array((-0.39866667, 6))
                else:
                    coeffs = np.loadtxt(csv_file_path)
                mv_val = (eV_val - coeffs[1]) / coeffs[0]
            if 'Zeiss' in self.stack_meta['img1']['System']['SystemType']:
                mv_val = self.stack_meta[page]['TLD']['Mirror']   # NOTE True? Assuming that Zeiss ESB can be considered equivalent to TLD Mirror #
                # coeffs= np.array(1) # Placeholder - need proper calibration?
                # eht_target = self.stack_meta[page]['Beam']['HV']
                # eV_val = np.array(eht_target - (ana_voltage*coeffs))
                eV_val = mv_val
            labels.append(rf'TLD_Mirror{i + 1}_' + str(mv_val) + '.tif')
        tf.imwrite(rf"{esb_path}{slash}esb_stack.tif", self.stack, dtype=self.dtype_info.dtype, photometric='minisblack', imagej=True, resolution=(1. / pixel_width_um, 1. / pixel_width_um), metadata={'spacing': 1, 'unit': 'um', 'axes': 'ZYX', 'Labels': labels})
        with open(rf"{meta_path}{slash}esb_stack_meta.json", 'w') as f:
            json.dump(self.stack_meta, f, indent=2)
        if '_AC' in self.name and hasattr(self, 'stack_meta_r'):
            with open(rf"{meta_path}{slash}esb_stack_meta_r.json", 'w') as f:
                json.dump(self.stack_meta_r, f, indent=2)
        data.plot_stack_meta(self, reg, save_path=rf"{meta_path}{slash}_stack_meta_plots.png")
        plot_scalebar(data.img_avg(self), stack_meta=self.stack_meta, save_path=rf'{save_path}{slash}{self.name}_avg_img_scaled.png')
        readable_meta = {}
        for key, metadata in self.stack_meta.items():
            readable_meta[key] = {
                "Voltage_V": metadata['TLD']['Mirror'],
                "PixelWidth_um": metadata['Scan']['PixelWidth'] * 1e6,
                "PixelHeight_um": metadata['Scan']['PixelHeight'] * 1e6,
                "HorFieldsize_um": metadata['Scan']['HorFieldsize'] * 1e6,
                "VerFieldsize_um": metadata['Scan']['VerFieldsize'] * 1e6,
                "BeamCurrent_A": metadata['EBeam']['BeamCurrent'],
                "HV_kV": metadata['Beam']['HV'] / 1000.0,
                "Dwell_s": metadata['Scan']['Dwelltime'],
                "WD_mm": metadata['Stage']['WorkingDistance'] * 1e3,
                "File_ESB": metadata.get('Processing', {}).get('file_esb', metadata.get('Processing', {}).get('file')),
                "File_InLens": metadata.get('Processing', {}).get('file_inlens', None),
                "Shift_x": metadata.get('Processing', {}).get('transformation', {}).get('x', None),
                "Shift_y": metadata.get('Processing', {}).get('transformation', {}).get('y', None),
            }
        with open(rf"{meta_path}{slash}esb_stack_meta_readable.json", 'w') as f:
            json.dump(readable_meta, f, indent=2)
        try:
            metadata.metadata_params(self, write=True, out_folder_override=True)
        except Exception as e:
            pass
