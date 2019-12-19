import numpy as np
from PIL import Image

def posFunc_Vertical_Down(frame_id, frame_size=20, frame_shape=(64,64), patch_shape=(28,28)):
    h_step_size = (frame_shape[0] - patch_shape[0]) / frame_size
    v_step_size = (frame_shape[1] - patch_shape[1]) / frame_size
    
    init_pos = [(frame_shape[0] - patch_shape[0]) / 2, 0]
    
    if frame_id == 0:
        # np.random.randn: Normalized Gaussian Random Values
        pos = np.array(init_pos) + np.random.randn(2) 
    else:
        diffs = np.array([0, v_step_size*frame_id])
        pos = np.array(init_pos) + np.random.randn(2) + diffs
    return tuple(pos.astype(np.int8))


def posFunc_Vertical_Up(frame_id, frame_size=20, frame_shape=(64,64), patch_shape=(28,28)):
    h_step_size = (frame_shape[0] - patch_shape[0]) / frame_size
    v_step_size = (frame_shape[1] - patch_shape[1]) / frame_size
    
    init_pos = [(frame_shape[0] - patch_shape[0]) / 2, frame_shape[1] - patch_shape[1]]
    
    if frame_id == 0:
        # np.random.randn: Normalized Gaussian Random Values
        pos = np.array(init_pos) + np.random.randn(2) 
    else:
        diffs = np.array([0, - v_step_size*frame_id])
        pos = np.array(init_pos) + np.random.randn(2) + diffs
    return tuple(pos.astype(np.int8))


def posFunc_Horizontal_Right(frame_id, frame_size=20, frame_shape=(64,64), patch_shape=(28,28)):
    """
    ----->
    """
    h_step_size = (frame_shape[0] - patch_shape[0]) / frame_size
    v_step_size = (frame_shape[1] - patch_shape[1]) / frame_size
    
    init_pos = [0, (frame_shape[1] - patch_shape[1]) / 2 ]
    
    if frame_id == 0:
        # np.random.randn: Normalized Gaussian Random Values
        pos = np.array(init_pos) + np.random.randn(2) 
    else:
        diffs = np.array([h_step_size*frame_id, 0])
        pos = np.array(init_pos) + np.random.randn(2) + diffs
    return tuple(pos.astype(np.int8))


def posFunc_Horizontal_Left(frame_id, frame_size=20, frame_shape=(64,64), patch_shape=(28,28)):
    """
    <-----
    """
    h_step_size = (frame_shape[0] - patch_shape[0]) / frame_size
    v_step_size = (frame_shape[1] - patch_shape[1]) / frame_size
    
    init_pos = [frame_shape[0] - patch_shape[0], (frame_shape[1] - patch_shape[1]) / 2 ]
    
    if frame_id == 0:
        # np.random.randn: Normalized Gaussian Random Values
        pos = np.array(init_pos) + np.random.randn(2) 
    else:
        diffs = np.array([- h_step_size*frame_id, 0])
        pos = np.array(init_pos) + np.random.randn(2) + diffs
    return tuple(pos.astype(np.int8))


def posFunc_Left2Right_Down(frame_id, frame_size=20, frame_shape=(64,64), patch_shape=(28,28)):
    h_step_size = (frame_shape[0] - patch_shape[0]) / frame_size
    v_step_size = (frame_shape[1] - patch_shape[1]) / frame_size
    
    if frame_id == 0:
        # np.random.randn: Normalized Gaussian Random Values
        pos = np.zeros(2) + np.random.randn(2) 
    else:
        diffs = np.array([h_step_size*frame_id, v_step_size*frame_id])
        pos = np.zeros(2) + np.random.randn(2) + diffs
    return tuple(pos.astype(np.int8))


def posFunc_Left2Right_Up(frame_id, frame_size=20, frame_shape=(64,64), patch_shape=(28,28)):
    h_step_size = (frame_shape[0] - patch_shape[0]) / frame_size
    v_step_size = (frame_shape[1] - patch_shape[1]) / frame_size
    
    init_pos = [0, frame_shape[1] - patch_shape[1]]
    
    if frame_id == 0:
        # np.random.randn: Normalized Gaussian Random Values
        pos = np.array(init_pos) + np.random.randn(2) 
    else:
        diffs = np.array([h_step_size*frame_id, - v_step_size*frame_id])
        pos = np.array(init_pos) + np.random.randn(2) + diffs
    return tuple(pos.astype(np.int8))


def posFunc_Right2Left_Down(frame_id, frame_size=20, frame_shape=(64,64), patch_shape=(28,28)):
    h_step_size = (frame_shape[0] - patch_shape[0]) / frame_size
    v_step_size = (frame_shape[1] - patch_shape[1]) / frame_size
    
    init_pos = [frame_shape[0] - patch_shape[0], 0]
    
    if frame_id == 0:
        # np.random.randn: Normalized Gaussian Random Values
        pos = np.array(init_pos) + np.random.randn(2) 
    else:
        diffs = np.array([- h_step_size*frame_id, v_step_size*frame_id])
        pos = np.array(init_pos) + np.random.randn(2) + diffs
    return tuple(pos.astype(np.int8))


def posFunc_Right2Left_Up(frame_id, frame_size=20, frame_shape=(64,64), patch_shape=(28,28)):
    h_step_size = (frame_shape[0] - patch_shape[0]) / frame_size
    v_step_size = (frame_shape[1] - patch_shape[1]) / frame_size
    
    init_pos = [frame_shape[0] - patch_shape[0], frame_shape[1] - patch_shape[1]]
    
    if frame_id == 0:
        # np.random.randn: Normalized Gaussian Random Values
        pos = np.array(init_pos) + np.random.randn(2) 
    else:
        diffs = np.array([- h_step_size*frame_id, - v_step_size*frame_id])
        pos = np.array(init_pos) + np.random.randn(2) + diffs
    return tuple(pos.astype(np.int8))



def posFunc(frame_id, motion_label):
    if motion_label == 0:
        pos = posFunc_Vertical_Down(frame_id)
    elif motion_label == 1:
        pos = posFunc_Vertical_Up(frame_id)
    elif motion_label == 2:
        pos = posFunc_Horizontal_Left(frame_id)
    elif motion_label == 3:
        pos = posFunc_Horizontal_Right(frame_id)
    elif motion_label == 4:
        pos = posFunc_Left2Right_Down(frame_id)
    elif motion_label == 5:
        pos = posFunc_Left2Right_Up(frame_id)
    elif motion_label == 6:
        pos = posFunc_Right2Left_Down(frame_id)
    elif motion_label == 7:
        pos = posFunc_Right2Left_Up(frame_id)
    return pos