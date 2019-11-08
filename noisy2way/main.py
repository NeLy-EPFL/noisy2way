import os
import shutil
import tempfile
import pickle

from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from skimage import io
import tensorflow as tf


def train(stack, basedir, model_name):
    """
    Train the n2v network.

    Parameters
    ----------
    stack : numpy array
        A stack of images. First dimension time.
        Second and third dimension spatial.
    basedir : string
        Directory where output models are saved.
    model_name : string
        Name of the model that is trained.

    Returns
    -------
    model : keras model
        Trained model.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        os.mkdir(f"{tmpdir}/data")
        os.mkdir(f"{tmpdir}/data/validation")
        os.mkdir(f"{tmpdir}/data/train") 
        if stack.shape[0] < 100:
            raise ValueError("The stack needs to have at least 100 frames.")
        for i, img in enumerate(stack[:50:5]):
            io.imsave(f"{tmpdir}/data/train/img_{i}.tif", img)
        
        for i, img in enumerate(stack[50:100]):
            io.imsave(f"{tmpdir}/data/validation/img_{i}.tif", img)
        
        # We create our DataGenerator-object.
        # It will help us load data and extract patches for training and validation.
        datagen = N2V_DataGenerator()
        
        # We load all the '.tif' files from the 'data' directory.
        # If you want to load other types of files see the RGB example.
        # The function will return a list of images (numpy arrays).
        train_imgs = datagen.load_imgs_from_directory(directory=f"{tmpdir}/data/train")
        validation_imgs = datagen.load_imgs_from_directory(directory=f"{tmpdir}/data/validation")
        
        X = datagen.generate_patches_from_list(
            train_imgs, shape=(96, 96), shuffle=True, augment=False
        )
        X_val = datagen.generate_patches_from_list(
            validation_imgs, shape=(96, 96), augment=False
        )
        
        # You can increase "train_steps_per_epoch" to get even better results at the price of longer computation.
        config = N2VConfig(
            X,
            unet_kern_size=3,
            train_steps_per_epoch=50,
            train_epochs=40,
            train_loss="mse",
            batch_norm=True,
            train_batch_size=128,
            n2v_perc_pix=1.6,
            n2v_patch_shape=(64, 64),
            n2v_manipulator="uniform_withCP",
            n2v_neighborhood_radius=5,
        )
        
        # a name used to identify the model
        #model_name = "n2v_2D_shift"
        # the base directory in which our model will live
        #basedir = "models"
        # We are now creating our network model.
        basedir = os.path.abspath(os.path.expanduser(os.path.expandvars(basedir)))
        model = N2V(config, model_name, basedir=basedir)
        history = model.train(X, X_val)
        with open(os.path.join(basedir, f"{model_name}/history.pkl"), "wb") as fp:
            pickle.dump(history, fp)
        return model


def predict(model, stack):
    """
    Predicts denoised frames based on a given model.

    Parameters
    ----------
    model : keras model
        Model returned by function::`train`.
    stack : 3D numpy array
        Stack of frames that should be denoised.
        First dimension is time. Second and third dimension are
        spatial.

    Returns
    -------
    results_stack : 3D numpy array
        Stack of denoised images.
    """
    stack = stack.astype(np.float32)
    results_stack = np.zeros_like(stack)
    
    for i, img in enumerate(stack):
            results_stack[i] = model.predict(img, axes='YX', n_tiles=(2,1))
    
    return results_stack

def find_shift(stack):
    """
    Finds the appropriate two way alignment shift for a stack.
    The stack is separated into even and odd rows. The shift is then
    determined by comparing the l1 norm of the difference between the
    two halfs of the image for different displacements.

    Parameters
    ----------
    stack : 3D numpy array
        Stack of frames that should be corrected.
        First dimension is time. Second and third dimension are
        spatial.
   
    Returns
    -------
    shift : int
        Optimal shift.
    """
    half0 = stack[:, ::2, :]
    half1 = stack[:, 1::2, :]
    
    assert half0.shape == half1.shape
    
    max_shift = 20
    l1_errors = np.zeros(2 * max_shift + 1)
    l2_errors = np.zeros(2 * max_shift + 1)
    shift_values = list(range(-max_shift, max_shift + 1))
    
    for i, shift in enumerate(shift_values):
        diff = np.zeros((half1.shape[0], half1.shape[0] - abs(shift)))
        if shift < 0:
            diff = half1[:, :, abs(shift):] - half0[:, :, :shift]
        elif shift == 0:
            diff = half1 - half0
        else:
            diff = half1[:, :, :-shift] - half0[:, :, shift:]
        l2 = np.sum(diff ** 2)
        l1 = np.sum(np.abs(diff))
        l1_errors[i] = l1
        l2_errors[i] = l2
    return shift_values[np.argmin(l1_errors)]


def apply_shift(stack, shift):
    """
    Applies a shift value to a stack.

    Parameters
    ----------
    stack : 3D numpy array
        Stack of frames that should be corrected by shift.
        First dimension is time. Second and third dimension are
        spatial.
    shift : int
        Shift magnitude.

    Returns
    -------
    stack : 3D numpy array
        Stack of frames after applying shift.
        First dimension is time. Second and third dimension are
        spatial.
    """
    if shift < 0:
        off_set = 1
        shift = abs(shift)
    else:
        off_set = 0
    print(off_set, shift)
    stack[:, off_set::2, :-shift] = stack[:, off_set::2, shift:]
    return stack


def correct_2way_alignment(stack, basedir, model_name):
    """
    Corrects two way alignment in a given stack.

    Parameters
    ----------
    stack : numpy array
        A stack of images. First dimension time.
        Second and third dimension spatial.
    basedir : string
        Directory where output models are saved.
    model_name : string
        Name of the model that is trained.

    Returns
    -------
    shifted : 3D numpy array
        Stack after correcting two way alignment
    """
    model = train(stack, basedir, model_name)
    denoised = predict(model, stack)
    shift = find_shift(denoised)
    if shift == 0:
        print("No shift detected, returning identity!")
        return stack
    shifted = apply_shift(stack, shift)
    model_after_shift = train(shifted, basedir, model_name + "_retrained")
    denoised_after_shift = predict(model_after_shift, shifted)
    new_shift = find_shift(denoised_after_shift)
    if int(new_shift) == 0:
        print(f'Detected a shift of {shift}.')
        return shifted
    else:
        raise RuntimeError(f"Could not find appropriate shift value. Original shift: {shift}, New shift: {new_shift}")