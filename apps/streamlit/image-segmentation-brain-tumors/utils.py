import io
import matplotlib as mpl
import os
import streamlit as st
import zipfile

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from variables import data_path


def init_session_state_variables():
    """
    Initialize session state variables which manages the operation of the application
    """
    if 'pred_seg' not in st.session_state:
        # Save the predicted segmentation, if generated
        st.session_state["pred_seg"] = None

        # Save the selected patient's path (to get his images)
        st.session_state["patient_path"] = None

        # Token which indicates if user has changed the patient. If this is the case, we must remove the previous predicted segmentation, and generate the new patient's path
        st.session_state["patient_has_changed"] = True

        # Token to know if we can display the predicted segmentation or not (selected patient changed case)
        st.session_state["pred_can_be_displayed"] = False

        # Token to know if user has selected a new patient before predict (to avoid re-predicting an already predicted segmentation)
        st.session_state["pred_gen_for_this_patient"] = False


def get_key_from_dict(modality_dict, val):
    """
    Link modalities to their file names
    :param modality_dict: Dictionary which links the modalities to their file names in the database
    :param val: Name of a modality
    :return: Linked path to this modality
    """
    for key, value in modality_dict.items():
        if val == value:
            return key


def dataset_unzip():
    """
    Unzip the dataset if not already done
    """
    # Local usage
    # path_to_zip_file = "brats20-dataset-training-validation.zip"
    # target_dir = "brats20-dataset-training-validation"

    # AI Deploy usage
    path_to_zip_file = "/workspace/BraTS2020_dataset_zip/brats20-dataset-training-validation.zip"

    target_dir = "/workspace/brats20-dataset-training-validation"
    # /workspace/BraTS2020_zip_dataset/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData

    # Temporarily displays a spinner while unzipping the dataset
    with st.spinner("Unzipping dataset..."):
        # Check if target_dir already exists (already unzipped from a previous run)
        if not os.path.exists(target_dir):
            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(target_dir)


def rename_wrong_file(dataset_path):
    """
    Rename the incorrectly named file in the dataset
    :param dataset_path: Dataset's path
    """
    # Absolute path of the wrong file file
    old_name = dataset_path + "/BraTS20_Training_355/W39_1998.09.19_Segm.nii"
    new_name = dataset_path + "/BraTS20_Training_355/BraTS20_Training_355_seg.nii"

    # Renaming the file only if has not already been done
    if os.path.exists(old_name):
        os.rename(old_name, new_name)


def check_if_dataset_exists():
    """
    Check if the dataset exists in the environment to know if we can launch the app
    """
    # Check if dataset folder exists
    if not os.path.exists(data_path):
        st.error(f"Error: Dataset not found at {data_path}. App cannot be launched without data")
        st.stop()


def create_colormap():
    """
    Create a custom colormap and a norm to fix the color of each class
    Create a legend to know which color represents which class
    :return: Colormap norm & legend elements
    """
    my_cmap = mpl.colors.ListedColormap(['#440054', '#3b528b', '#18b880', '#e6d74f'])
    my_norm = mpl.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], my_cmap.N)
    class_names = ['class 0', 'class 1', 'class 2', 'class 3']
    legend = [plt.Rectangle((0, 0), 1, 1, color=my_cmap(i), label=class_names[i]) for i in range(len(class_names))]
    return my_cmap, my_norm, legend


def download_file(fig, selected_sample, selected_plane, selected_slice):
    """
    Download generated figures as .png thanks to a download button. Filename will be based on the following info
    :param selected_sample: Patient number (str)
    :param selected_plane: Plane of visualization (str)
    :param selected_slice: Slice number (int)
    :param fig: generated figure
    """
    # Convert the figure to a file-like object
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # Create the download button with the file data and some metadata
    st.download_button(
        label='Download plot',
        data=buf,
        file_name=f"patient{selected_sample[-3:]}_{selected_plane}_slice{selected_slice}.png",
        mime='image/png',
    )
