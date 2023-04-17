from variables import data_path, VOLUME_SLICES
import nibabel as nib
import numpy as np
import cv2
import random
import os
import streamlit as st

from UNet_2D import *


def get_selected_patient_path(samples_list, selected_sample):
    """
    Get the selected patient's path
    :param samples_list: List of test patients
    :param selected_sample: Patient selected by the user
    :return: path of the selected patient
    """
    if selected_sample == 'Random patient':
        # Choose a random patient
        selected_sample = random.choice(samples_list[1:])

    # Get path of this patient
    patient_path = os.path.join(data_path, selected_sample, selected_sample).replace("\\", "/")
    return patient_path


def predict_btn_click(model, patient_path):
    """
    Callback function that is called when the user click the predict a patient's segmentation button
    This function call the predict function, and updates some session state variables (tokens)
    :param model: Trained CNN
    :param patient_path: Path of the patient and his images
    """
    # Check if user has selected a new patient before predict (to avoid re-predicting an already predicted segmentation)
    if not st.session_state["pred_gen_for_this_patient"]:
        predicted_seg = predict_segmentation(model, patient_path)

        # Save the predicted segmentation (since we can not return it in a callback function)
        st.session_state["pred_seg"] = predicted_seg

        # Set token to true so we know we can display the predicted segmentation, as long as the patient does not change
        st.session_state["pred_can_be_displayed"] = True

        # Set token to true to know if the patient's segmentation has already been predicted or not
        st.session_state["pred_gen_for_this_patient"] = True


def predict_segmentation(model, patient_path):
    """
    Predict patient's segmentation
    :param model: Trained CNN
    :param patient_path: Path of the patient so we can reach his images (T1CE + FLAIR)
    :return: Predicted segmentation
    """
    # Load NIfTI (.nii) files of the sample (patient)
    t1ce = nib.load(patient_path + '_t1ce.nii').get_fdata()
    flair = nib.load(patient_path + '_flair.nii').get_fdata()

    # Perform the same operations as our DataGenerator, to keep the same input shape
    # Create an empty array which will contain all the slices of the 2 modalities, resized in a (128, 128) shape 
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))

    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j], (IMG_SIZE, IMG_SIZE))
        X[j, :, :, 1] = cv2.resize(t1ce[:, :, j], (IMG_SIZE, IMG_SIZE))

    # Send our images to the CNN model and return predicted segmentation
    return model.predict(X / np.max(X), verbose=0)


def patient_has_changed_update_token():
    """
    Callback function that is called when the user changes the selected patient from the st.selectbox() component
    This function updates some session state variables (tokens).
    """
    # If patient is changed, we update the token
    st.session_state["patient_has_changed"] = True

    # If patient is changed, we no longer display the previous prediction since it corresponds to another patient
    st.session_state["pred_can_be_displayed"] = False

    # Current patient's segmentation has not been predicted yet
    st.session_state["pred_gen_for_this_patient"] = False




