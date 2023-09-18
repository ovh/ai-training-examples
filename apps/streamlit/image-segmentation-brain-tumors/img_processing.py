import cv2
import numpy as np
import streamlit as st
from skimage.transform import rotate


def modality_and_ground_truth_processing(col1, visualization_plane, selected_modality, ground_truth_seg, modality):
    """
    Adapt modality & ground truth depending on the selected plane, so they can be correctly displayed
    :param col1: Column where we will display a slider component
    :param visualization_plane: plane (Axial / Coronal / Sagittal)
    :param selected_modality: Option selected by the user (T1 / T1CE / T2 / FLAIR)
    :param ground_truth_seg: Original segmentation, realized by experts
    :param modality: MRI image
    :return: adapted modality, ground_truth_seg, and the selected_slice number designating the slice that the user wishes to view
    """

    img_shape = ground_truth_seg.shape

    if visualization_plane == "Sagittal":
        with col1:
            # Display a slider for slice selection
            selected_slice = st.slider("Displayed slice", 0, img_shape[0] - 1, int(img_shape[0] / 2), help="Select displayed slice")

        if selected_modality is not None:
            modality = modality[selected_slice, :, :]
            modality = np.array(rotate(modality, 90, resize=True))

        ground_truth_seg = ground_truth_seg[selected_slice, :, :]
        ground_truth_seg = np.array(rotate(ground_truth_seg, 90, resize=True, order=0))

    elif visualization_plane == "Coronal":
        with col1:
            # Display a slider for slice selection
            selected_slice = st.slider("Displayed slice", 0, img_shape[1] - 1, int(img_shape[1] / 2), help="Select displayed slice")

        if selected_modality is not None:
            modality = modality[:, selected_slice, :]
            modality = np.array(rotate(modality, 90, resize=True))

        ground_truth_seg = ground_truth_seg[:, selected_slice, :]
        ground_truth_seg = np.array(rotate(ground_truth_seg, 90, resize=True, order=0))

    else:
        # visualization_plane == "Axial"
        with col1:
            # Display a slider for slice selection
            selected_slice = st.slider("Displayed slice", 0, img_shape[2] - 1, int(img_shape[2] / 2), help="Select displayed slice")

        if selected_modality is not None:
            modality = cv2.resize(modality[:, :, selected_slice], (128, 128),
                                  interpolation=cv2.INTER_NEAREST)
        ground_truth_seg = cv2.resize(ground_truth_seg[:, :, selected_slice], (128, 128),
                                      interpolation=cv2.INTER_NEAREST)

    return modality, ground_truth_seg, selected_slice


def resize_predicted_seg(pred_seg):
    """
    Resize the predicted segmentation from (155, 128, 128, 4) to (155, 240, 240, 4) to give it the same size as the displayed modality
    :param pred_seg: predicted segmentation
    :return: resized predicted segmentation
    """
    resized_img = np.zeros((155, 240, 240, 4))
    for i in range(155):
        for j in range(4):
            resized_img[i, :, :, j] = cv2.resize(pred_seg[i, :, :, j], (240, 240), interpolation=cv2.INTER_NEAREST)
    pred_seg = resized_img
    return pred_seg


def predicted_seg_processing(pred_seg, visualization_plane, selected_slice, displayed_class, post_processing_token):
    """
    Adapt the predicted segmentation according to the parameters chosen by the user (plane, slice, class, post-processing)
    :param pred_seg: Patient's predicted segmentation
    :param visualization_plane: plane (Axial / Coronal / Sagittal)
    :param selected_slice: Integer number indicating the slice to be displayed [0;154] / [0;239] depending on the chosen plane
    :param displayed_class: All / Background (0) / 1 / 2 / 3
    :param post_processing_token: post_processing checkbox value
    :return: Adapted predicted segmentation
    """
    if visualization_plane == "Sagittal":
        pred_seg = resize_predicted_seg(pred_seg)

        if post_processing_token:
            # Obtain the class that has obtained the highest probability for each pixel to get same colors as in ground truth display
            pred_seg = np.argmax(pred_seg, axis=3)
            pred_seg = pred_seg[:, selected_slice, :]

            # Keep only selected class
            pred_seg = pred_seg.astype(float)
            pred_seg = display_specific_class(pred_seg, displayed_class)

        else:
            pred_seg = pred_seg[:, selected_slice, :, 1:4]

        # Flip the predicted segmentation so that it is displayed correctly
        pred_seg = cv2.flip(pred_seg, 0)

    elif visualization_plane == "Coronal":
        pred_seg = resize_predicted_seg(pred_seg)

        if post_processing_token:
            # Obtain the class that has obtained the highest probability for each pixel to get same colors as in ground truth display
            pred_seg = np.argmax(pred_seg, axis=3)
            pred_seg = pred_seg[:, :, selected_slice]

            # Keep only selected class
            pred_seg = pred_seg.astype(float)
            pred_seg = display_specific_class(pred_seg, displayed_class)

        else:
            pred_seg = pred_seg[:, :, selected_slice, 1:4]

        # Flip the predicted segmentation so that it is displayed correctly
        pred_seg = cv2.flip(pred_seg, 0)

    else:
        # visualization_plane == "Axial"
        if post_processing_token:
            # Obtain the class that has obtained the highest probability for each pixel to get same colors as in ground truth display
            pred_seg = np.argmax(pred_seg, axis=3)
            pred_seg = pred_seg[selected_slice, :, :]

            # Keep only selected class
            pred_seg = pred_seg.astype(float)
            pred_seg = display_specific_class(pred_seg, displayed_class)

        else:
            pred_seg = pred_seg[selected_slice, :, :, 1:4]

    return pred_seg


def display_specific_class(image, class_mode):
    """
    Visualize a single class selected by the user, by setting all others to np.nan
    :param image: Segmentation image (original or predicted)
    :param class_mode: Class label (All, 0, 1, 2, 3)
    :return: updated segmentation
    """
    # All classes except background
    if class_mode == "All":
        image[image == 0] = np.nan

    # Not tumor (Healthy zone / Background)
    elif class_mode == "0 - (Not Tumor)":
        image[image != 0] = np.nan

    # Necrotic and Non-Enhancing tumor
    elif class_mode == "1 - (Non-Enhancing Tumor)":
        image[image != 1] = np.nan

    # Peritumoral Edema
    elif class_mode == "2 - (Peritumoral Edema)":
        image[image != 2] = np.nan

    # Enhancing tumor
    elif class_mode == "3 - (Enhancing Tumor)":
        image[image != 3] = np.nan

    elif class_mode == "None":
        image = None

    return image
