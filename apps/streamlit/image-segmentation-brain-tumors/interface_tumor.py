import matplotlib.pyplot as plt
import nibabel as nib
import streamlit as st

from img_processing import modality_and_ground_truth_processing, predicted_seg_processing, display_specific_class
from variables import modalities_dict, samples_test
from predict_seg import patient_has_changed_update_token, get_selected_patient_path, predict_btn_click
from utils import get_key_from_dict, create_colormap, download_file


def launch_app(model):
    # Define columns to place app components
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        # Select a patient among the test patients list
        selected_sample = st.selectbox('Select a test patient', samples_test,
                                       help="Select one of the patient from the test dataset",
                                       on_change=patient_has_changed_update_token)

        # If user selects a new patient, we get & save his path
        if st.session_state["patient_has_changed"]:
            patient_path = get_selected_patient_path(samples_test, selected_sample)
            st.session_state["patient_has_changed"] = False
            st.session_state["patient_path"] = patient_path

    with col2:
        # Select a visualization plane & display a checkbox to enable/disable post-processing
        selected_plane = st.selectbox("Plane", ["Axial", "Coronal", "Sagittal"], help="Select plane of visualization")
        st.write("Post-Processing")
        post_processing = st.checkbox("Enable", help="Get same colors for prediction by applying an argmax function")

    with col3:
        # Select a modality that will be displayed behind the segmentations (ground truth + prediction)
        selected_options = st.selectbox("Background View", ["None", "T1", "T1CE", "T2", "FLAIR"],
                                        help="Select displayed modality", index=1)
        displayed_class = st.selectbox("Displayed class",
                                       ["None", "All", "0 - (Not Tumor)", "1 - (Non-Enhancing Tumor)",
                                        "2 - (Peritumoral Edema)", "3 - (Enhancing Tumor)"], index=1, help="Select displayed class")

    with col4:
        # Add a button so the user can predict patient's segmentation
        st.write("")
        st.write("")
        predict_btn = st.button("Predict patient's tumor segmentation", on_click=predict_btn_click,
                                args=(model, st.session_state["patient_path"]))

    # Get modality filename depending on the selected_options by the user (T1, T1CE, T2, FLAIR)
    selected_modality = get_key_from_dict(modalities_dict, selected_options)

    # Load patient's images
    # 1 - Ground truth
    ground_truth_seg_modality = nib.load(st.session_state["patient_path"] + "_seg.nii").get_fdata()

    # 2 - Modality (If there is a background view (not None))
    if selected_modality is not None:
        # Load background modality
        displayed_modality = nib.load(st.session_state["patient_path"] + selected_modality).get_fdata()
    else:
        displayed_modality = None

    # Adapt modality & ground truth depending on the selected plane, so they can be correctly displayed
    displayed_modality, ground_truth_seg_modality, selected_slice = modality_and_ground_truth_processing(col1,
                                                                                                         selected_plane,
                                                                                                         selected_modality,
                                                                                                         ground_truth_seg_modality,
                                                                                                         displayed_modality)

    # Fix 4 to 3 since it is missing, for colors display purposes (to match the values of the predicted segmentation)
    ground_truth_seg_modality[ground_truth_seg_modality == 4] = 3

    # Keep only desired classes
    ground_truth_seg_modality = display_specific_class(ground_truth_seg_modality, displayed_class)

    # Adapt predicted segmentation (if user has predicted something)
    if st.session_state["pred_seg"] is not None:
        pred_seg = st.session_state["pred_seg"]
        pred_seg = predicted_seg_processing(pred_seg, selected_plane, selected_slice, displayed_class, post_processing)
    else:
        pred_seg = None

    # Define new columns to display our images
    col5, col6, col7, col8 = st.columns([1, 1, 1, 1])

    # Create colormap
    cmap, norm, legend = create_colormap()

    # Display of ground truth above modality (if one selected)
    with col5:
        fig1, ax1 = plt.subplots()
        if selected_modality is not None:
            ax1.imshow(displayed_modality, cmap="gray")
        if ground_truth_seg_modality is not None:
            ax1.imshow(ground_truth_seg_modality, alpha=0.7, cmap=cmap, norm=norm)
            ax1.legend(handles=legend, loc='upper right')
        ax1.axis('off')
        ax1.set_title("Ground truth segmentation", y=0.01, color="white")

        st.pyplot(fig1)

        # Add a download figure button
        download_file(fig1, st.session_state["patient_path"], selected_plane, selected_slice)

    # Display of predicted segmentation above modality (if one selected)

    # If user hasn't changed the selected patient, we display the predicted segmentation.
    # If he selects an other patient, the old prediction should no longer be displayed. The user must generate a new one by clicking click the predict button.
    with col6:
        if st.session_state["pred_can_be_displayed"]:
            fig2, ax2 = plt.subplots()
            if selected_modality is not None:
                ax2.imshow(displayed_modality, cmap="gray")
            if pred_seg is not None:
                if post_processing:
                    ax2.imshow(pred_seg, alpha=0.7, cmap=cmap, norm=norm)
                    # Legend is only adapted for a post-processed predicted segmentation
                    ax2.legend(handles=legend, loc='upper right')
                else:
                    ax2.imshow(pred_seg, alpha=0.7)
            ax2.axis('off')
            ax2.set_title("Predicted segmentation", y=0.01, color="white")

            st.pyplot(fig2)

            # Add a download figure option
            download_file(fig2, st.session_state["patient_path"], selected_plane, selected_slice)

