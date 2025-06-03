import streamlit as st
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import nibabel as nib
import subprocess 
from pathlib import Path
import sys
import configparser

def install_pyalfe():
    """
    Install PyALFE if not already installed.
    Returns True if successful, False otherwise.
    """
    try:
        # Check if pyalfe is already installed
        import pyalfe
        return True
    except ImportError:
        pass
    
    # Check if pyalfe directory exists
    pyalfe_dir = os.path.join(os.getcwd(), 'pyalfe')
    if not os.path.exists(pyalfe_dir):
        st.error(f"PyALFE directory not found at: {pyalfe_dir}")
        st.info("Please clone PyALFE: `git clone https://github.com/reghbali/pyalfe.git`")
        return False
    
    try:
        st.info("Installing PyALFE... This may take a few minutes.")
        
        # Change to pyalfe directory and install
        original_cwd = os.getcwd()
        os.chdir(pyalfe_dir)
        
        # Install build tools
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "build"], 
                      check=True, capture_output=True)
        
        # Build the package
        subprocess.run([sys.executable, "-m", "build"], 
                      check=True, capture_output=True)
        
        # Find the built wheel file
        dist_dir = os.path.join(pyalfe_dir, "dist")
        wheel_files = [f for f in os.listdir(dist_dir) if f.endswith('.whl')]
        if not wheel_files:
            raise Exception("No wheel file found after build")
        
        wheel_file = os.path.join(dist_dir, wheel_files[0])
        
        # Install the wheel
        subprocess.run([sys.executable, "-m", "pip", "install", wheel_file], 
                      check=True, capture_output=True)
        
        # Download models
        subprocess.run([sys.executable, "-c", "import pyalfe; pyalfe.download('models')"], 
                      check=True, capture_output=True)
        
        os.chdir(original_cwd)
        st.success("PyALFE installed successfully!")
        return True
        
    except Exception as e:
        os.chdir(original_cwd)
        st.error(f"Failed to install PyALFE: {str(e)}")
        return False

def update_existing_config(input_dir, output_dir, config_path="/workspaces/ALD-Loes-score/pyalfe/config_ald.ini"):
    """
    Update the existing config file with new input and output directories.
    
    Args:
        input_dir: Path to the input directory containing the accessions
        output_dir: Path to the output directory for processed results
        config_path: Path to the existing config file to update
    """
    try:
        # Read the existing config file
        config = configparser.ConfigParser()
        
        # Check if config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
        config.read(config_path)
        
        # Update only the input_dir and output_dir in the options section
        if 'options' not in config:
            config['options'] = {}
        
        config['options']['input_dir'] = input_dir
        config['options']['output_dir'] = output_dir
        
        # Write the updated config back to the file
        with open(config_path, 'w') as configfile:
            config.write(configfile)
        
        return config_path
        
    except Exception as e:
        raise Exception(f"Failed to update config file: {str(e)}")

def load_and_store_dicom_series(directory, session_key):
    if session_key not in st.session_state:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        reader.SetFileNames(dicom_names)
        image_sitk = reader.Execute()
        image_np = sitk.GetArrayFromImage(image_sitk)
        st.session_state[session_key] = image_np
    return st.session_state[session_key]

def plot_slice(slice, size=(4, 4), is_nifti=False):
    # Adjust the figure size for consistent viewer sizes
    fig, ax = plt.subplots(figsize=size)
    # Calculate the square canvas size
    canvas_size = max(slice.shape)
    canvas = np.full((canvas_size, canvas_size), fill_value=slice.min(), dtype=slice.dtype)
    # Center the image within the canvas
    x_offset = (canvas_size - slice.shape[0]) // 2
    y_offset = (canvas_size - slice.shape[1]) // 2
    canvas[x_offset:x_offset+slice.shape[0], y_offset:y_offset+slice.shape[1]] = slice
    fig.patch.set_facecolor('black')  # Set the figure background to black
    ax.set_facecolor('black')
    if is_nifti:
        canvas = np.rot90(canvas)
    else:
        canvas = canvas[::-1, ::-1]

    ax.imshow(canvas, cmap='gray')
    ax.axis('off')
    return fig

def load_nifti_file(filepath, session_key):
    if session_key not in st.session_state:
        try:
            # First try to load as-is
            nifti_img = nib.load(filepath)
            image_np = np.asanyarray(nifti_img.dataobj)
            st.session_state[session_key] = image_np
        except Exception as e:
            # If it fails, try different approaches
            st.error(f"Error loading NIfTI file: {str(e)}")
            
            # Try to determine the actual file type
            try:
                with open(filepath, 'rb') as f:
                    header = f.read(10)
                    
                if header.startswith(b'\x1f\x8b'):
                    st.error("File appears to be gzipped but nibabel couldn't load it. The file might be corrupted.")
                elif header[:4] == b'\x5c\x01\x00\x00' or header[:4] == b'\x00\x00\x01\x5c':
                    st.error("File appears to be a NIfTI file but couldn't be loaded. Try saving as uncompressed .nii format.")
                else:
                    st.error(f"File doesn't appear to be a valid NIfTI file. Header: {header}")
                    
            except Exception as header_error:
                st.error(f"Couldn't analyze file header: {str(header_error)}")
            
            # Return a dummy array to prevent further errors
            st.session_state[session_key] = np.zeros((100, 100, 50))
            
    return st.session_state[session_key]

def create_pyalfe_directory_structure(flair_file, t1_file, base_temp_dir, accession_name="ACCESSION"):
    """
    Create the directory structure expected by PyALFE.
    
    Args:
        flair_file: Streamlit uploaded FLAIR file
        t1_file: Streamlit uploaded T1 file  
        base_temp_dir: Base temporary directory
        accession_name: Name for the accession directory
    
    Returns:
        str: Path to the accession directory
    """
    
    # Create the main structure
    accession_dir = os.path.join(base_temp_dir, accession_name)
    
    # Create subdirectories
    flair_dir = os.path.join(accession_dir, "FLAIR")
    t1_dir = os.path.join(accession_dir, "T1")
    
    os.makedirs(flair_dir, exist_ok=True)
    os.makedirs(t1_dir, exist_ok=True)
    
    # Save FLAIR file
    if flair_file:
        # Determine the correct extension based on the uploaded filename
        original_name = flair_file.name
        if original_name.endswith('.nii.gz'):
            flair_filename = "FLAIR.nii.gz"
        elif original_name.endswith('.nii'):
            flair_filename = "FLAIR.nii"
        else:
            # Default to .nii.gz
            flair_filename = "FLAIR.nii.gz"
            
        flair_path = os.path.join(flair_dir, flair_filename)
        with open(flair_path, 'wb') as f:
            f.write(flair_file.getvalue())
    
    # Save T1 file
    if t1_file:
        # Determine the correct extension based on the uploaded filename
        original_name = t1_file.name
        if original_name.endswith('.nii.gz'):
            t1_filename = "T1.nii.gz"
        elif original_name.endswith('.nii'):
            t1_filename = "T1.nii"
        else:
            # Default to .nii.gz
            t1_filename = "T1.nii.gz"
            
        t1_path = os.path.join(t1_dir, t1_filename)
        with open(t1_path, 'wb') as f:
            f.write(t1_file.getvalue())
    
    return accession_dir

def run_pyalfe_on_directory(accession_dir, temp_base_dir):
    """
    Run PyALFE on a directory structure containing MRI data.
    
    Args:
        accession_dir: Path to the accession directory
        temp_base_dir: Base temporary directory for input/output
    
    Returns:
        dict: Results from PyALFE processing
    """
    try:
        import pyalfe
    except ImportError:
        return {
            "success": False,
            "error": "PyALFE not installed. Please install it first.",
            "output": None
        }
    
    # Check if accession directory exists
    if not os.path.exists(accession_dir):
        return {
            "success": False,
            "error": f"Accession directory not found at: {accession_dir}",
            "output": None
        }
    
    try:
        # Create output directory
        output_dir = os.path.join(temp_base_dir, "processed_ald")
        os.makedirs(output_dir, exist_ok=True)
        
        # Update the existing config file with temporary paths
        config_path = "/workspaces/ALD-Loes-score/pyalfe/config_ald.ini"
        input_dir = os.path.dirname(accession_dir)  # Parent directory containing the accession
        
        # Store original config values to restore later
        original_config = None
        try:
            original_config = configparser.ConfigParser()
            original_config.read(config_path)
            original_input_dir = original_config.get('options', 'input_dir', fallback=None)
            original_output_dir = original_config.get('options', 'output_dir', fallback=None)
        except:
            original_input_dir = None
            original_output_dir = None
        
        # Update config with temporary paths
        update_existing_config(input_dir, output_dir, config_path)
        
        # Get the accession name (last part of the path)
        accession_name = os.path.basename(accession_dir)
        
        # Change to the parent directory of the accession
        original_cwd = os.getcwd()
        parent_dir = os.path.dirname(accession_dir)
        os.chdir(parent_dir)
        
        try:
            # Run PyALFE with the accession name and updated config
            result = subprocess.run(
                ["pyalfe", "run", "-c", config_path, accession_name],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=parent_dir
            )
            
            # Restore original config values if they existed
            if original_input_dir and original_output_dir:
                try:
                    update_existing_config(original_input_dir, original_output_dir, config_path)
                except:
                    pass  # If restoration fails, continue anyway
            
            if result.returncode == 0:
                # Check if output files were created
                output_files = []
                if os.path.exists(output_dir):
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            output_files.append(os.path.join(root, file))
                
                return {
                    "success": True,
                    "output": result.stdout,
                    "error": None,
                    "stderr": result.stderr if result.stderr else None,
                    "output_files": output_files,
                    "output_dir": output_dir
                }
            else:
                return {
                    "success": False,
                    "error": f"PyALFE command failed with return code {result.returncode}",
                    "output": result.stdout,
                    "stderr": result.stderr
                }
        finally:
            os.chdir(original_cwd)
            
    except subprocess.TimeoutExpired:
        os.chdir(original_cwd)
        # Try to restore original config on timeout
        if original_input_dir and original_output_dir:
            try:
                update_existing_config(original_input_dir, original_output_dir, config_path)
            except:
                pass
        return {
            "success": False,
            "error": "PyALFE command timed out after 5 minutes",
            "output": None 
        }
    except Exception as e:
        os.chdir(original_cwd)
        # Try to restore original config on error
        if original_input_dir and original_output_dir:
            try:
                update_existing_config(original_input_dir, original_output_dir, config_path)
            except:
                pass
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "output": None
        }

def process_uploaded_files_with_pyalfe(flair_file, t1_file):
    """
    Process uploaded FLAIR and T1 files with PyALFE by creating the expected directory structure.
    
    Args:
        flair_file: Streamlit uploaded FLAIR file
        t1_file: Streamlit uploaded T1 file
    
    Returns:
        dict: Results from PyALFE processing
    """
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create the PyALFE directory structure
            accession_dir = create_pyalfe_directory_structure(flair_file, t1_file, temp_dir)
            
            # Run PyALFE on the directory with dynamic config
            result = run_pyalfe_on_directory(accession_dir, temp_dir)
            return result
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process files: {str(e)}",
                "output": None
            }

def main():
    st.set_page_config(page_title='ALD Loes Score Calculator', layout="wide")

    st.title("ALD Loes Score Calculator")

    # Check PyALFE installation status
    if 'pyalfe_installed' not in st.session_state:
        st.session_state.pyalfe_installed = False
        try:
            import pyalfe
            st.session_state.pyalfe_installed = True
        except ImportError:
            pass

    # Custom CSS to style the "cards" with unique colors for each view
    st.markdown("""
    <style>
    .view-label-axial {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        background-color: #007BFF; /* Blue */
        border-radius: 10px;
        text-align: center;
        margin: 10px 0px;
        padding: 5px;
    }
    .view-label-coronal {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        background-color: #28A745; /* Green */
        border-radius: 10px;
        text-align: center;
        margin: 10px 0px;
        padding: 5px;
    }
    .view-label-sagittal {
        font-size: 16px;
        font-weight: bold;
        color: #ffffff;
        background-color: #ffc800; /* Yellow */
        border-radius: 10px;
        text-align: center;
        margin: 10px 0px;
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    # PyALFE Installation Section
    if not st.session_state.pyalfe_installed:
        st.warning("PyALFE is not installed. Install it to enable automated analysis.")
        if st.button("Install PyALFE", key="install_pyalfe"):
            if install_pyalfe():
                st.session_state.pyalfe_installed = True
                st.rerun()

    # File uploaders for FLAIR and T1
    st.subheader("Upload MRI Files")
    
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        st.markdown("**FLAIR Image**")
        flair_file = st.file_uploader(
            "Choose FLAIR NIfTI file", 
            type=["nii", "gz"], 
            key="flair_uploader",
            help="Upload your FLAIR (Fluid Attenuated Inversion Recovery) NIfTI file"
        )
    
    with col_upload2:
        st.markdown("**T1 Image**")
        t1_file = st.file_uploader(
            "Choose T1 NIfTI file", 
            type=["nii", "gz"], 
            key="t1_uploader",
            help="Upload your T1-weighted NIfTI file"
        )

    # Legacy uploader for backward compatibility
    st.markdown("---")
    st.markdown("**Or upload DICOM/NIfTI files (legacy mode)**")
    uploaded_files = st.file_uploader("Choose DICOM or NIfTI Files", accept_multiple_files=True, type=["dcm", "nii", "nii.gz"], key="file_uploader")
    
    # Determine which files to process
    files_to_process = None
    processing_mode = None
    
    if flair_file or t1_file:
        if flair_file and t1_file:
            files_to_process = {'flair': flair_file, 't1': t1_file}
            processing_mode = 'structured'
            st.success("✅ Both FLAIR and T1 files uploaded!")
        else:
            st.warning("⚠️ Please upload both FLAIR and T1 files for PyALFE analysis.")
            if flair_file:
                files_to_process = {'flair': flair_file}
                processing_mode = 'viewer_only'
            if t1_file:
                files_to_process = {'t1': t1_file}
                processing_mode = 'viewer_only'
    elif uploaded_files:
        files_to_process = uploaded_files
        processing_mode = 'legacy'
    
    if files_to_process:
        # Handle different processing modes
        if processing_mode == 'structured':
            # Load FLAIR and T1 for viewing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save files temporarily for viewing
                flair_path = os.path.join(temp_dir, 'flair.nii.gz')
                t1_path = os.path.join(temp_dir, 't1.nii.gz')
                
                with open(flair_path, 'wb') as f:
                    f.write(flair_file.getvalue())
                with open(t1_path, 'wb') as f:
                    f.write(t1_file.getvalue())
                
                # Load images for viewing
                flair_image = load_nifti_file(flair_path, "flair_image_data")
                t1_image = load_nifti_file(t1_path, "t1_image_data")
                
                # Display file information
                st.info(f"FLAIR image shape: {flair_image.shape}")
                st.info(f"T1 image shape: {t1_image.shape}")
                
                # Show FLAIR viewer
                st.subheader("FLAIR Image Viewer")
                col1, col2, col3 = st.columns(3)
                
                col1.markdown("<div class='view-label-axial'>Axial View (FLAIR)</div>", unsafe_allow_html=True)
                col2.markdown("<div class='view-label-coronal'>Coronal View (FLAIR)</div>", unsafe_allow_html=True)
                col3.markdown("<div class='view-label-sagittal'>Sagittal View (FLAIR)</div>", unsafe_allow_html=True)
                
                with col1:
                    flair_axial_slice = st.slider('FLAIR Axial', 0, flair_image.shape[2] - 1, flair_image.shape[2] // 2, key="flair_axial_slider")
                    fig = plot_slice(flair_image[:, :, flair_axial_slice], size=(3, 3), is_nifti=True)
                    st.pyplot(fig, clear_figure=True)

                with col2:
                    flair_coronal_slice = st.slider('FLAIR Coronal', 0, flair_image.shape[1] - 1, flair_image.shape[1] // 2, key="flair_coronal_slider")
                    fig = plot_slice(flair_image[:, flair_coronal_slice, :], size=(3, 3), is_nifti=True)
                    st.pyplot(fig, clear_figure=True)

                with col3:
                    flair_sagittal_slice = st.slider('FLAIR Sagittal', 0, flair_image.shape[0] - 1, flair_image.shape[0] // 2, key="flair_sagittal_slider")
                    fig = plot_slice(flair_image[flair_sagittal_slice, :, :], size=(3, 3), is_nifti=True)
                    st.pyplot(fig, clear_figure=True)
                
                # Show T1 viewer
                st.subheader("T1 Image Viewer")
                col1_t1, col2_t1, col3_t1 = st.columns(3)
                
                col1_t1.markdown("<div class='view-label-axial'>Axial View (T1)</div>", unsafe_allow_html=True)
                col2_t1.markdown("<div class='view-label-coronal'>Coronal View (T1)</div>", unsafe_allow_html=True)
                col3_t1.markdown("<div class='view-label-sagittal'>Sagittal View (T1)</div>", unsafe_allow_html=True)
                
                with col1_t1:
                    t1_axial_slice = st.slider('T1 Axial', 0, t1_image.shape[2] - 1, t1_image.shape[2] // 2, key="t1_axial_slider")
                    fig = plot_slice(t1_image[:, :, t1_axial_slice], size=(3, 3), is_nifti=True)
                    st.pyplot(fig, clear_figure=True)

                with col2_t1:
                    t1_coronal_slice = st.slider('T1 Coronal', 0, t1_image.shape[1] - 1, t1_image.shape[1] // 2, key="t1_coronal_slider")
                    fig = plot_slice(t1_image[:, t1_coronal_slice, :], size=(3, 3), is_nifti=True)
                    st.pyplot(fig, clear_figure=True)

                with col3_t1:
                    t1_sagittal_slice = st.slider('T1 Sagittal', 0, t1_image.shape[0] - 1, t1_image.shape[0] // 2, key="t1_sagittal_slider")
                    fig = plot_slice(t1_image[t1_sagittal_slice, :, :], size=(3, 3), is_nifti=True)
                    st.pyplot(fig, clear_figure=True)
        
        elif processing_mode == 'viewer_only':
            # Single file viewing mode
            with tempfile.TemporaryDirectory() as temp_dir:
                if 'flair' in files_to_process:
                    file_to_show = files_to_process['flair']
                    file_type = 'FLAIR'
                else:
                    file_to_show = files_to_process['t1']
                    file_type = 'T1'
                
                # Preserve original file extension
                original_name = file_to_show.name
                if original_name.endswith('.nii.gz'):
                    file_extension = '.nii.gz'
                elif original_name.endswith('.nii'):
                    file_extension = '.nii'
                else:
                    file_extension = '.nii.gz'  # default
                
                file_path = os.path.join(temp_dir, f'{file_type.lower()}{file_extension}')
                with open(file_path, 'wb') as f:
                    f.write(file_to_show.getvalue())
                
                # Display file info
                st.info(f"Loading {file_type} file: {original_name}")
                
                try:
                    image_np = load_nifti_file(file_path, f"{file_type.lower()}_image_data")
                    
                    # Only show viewer if loading was successful (check if it's not the dummy array)
                    if image_np.shape != (100, 100, 50):
                        st.subheader(f"{file_type} Image Viewer")
                        col1, col2, col3 = st.columns(3)
                        
                        col1.markdown(f"<div class='view-label-axial'>Axial View ({file_type})</div>", unsafe_allow_html=True)
                        col2.markdown(f"<div class='view-label-coronal'>Coronal View ({file_type})</div>", unsafe_allow_html=True)
                        col3.markdown(f"<div class='view-label-sagittal'>Sagittal View ({file_type})</div>", unsafe_allow_html=True)
                        
                        with col1:
                            axial_slice_num = st.slider(f'{file_type} Axial', 0, image_np.shape[2] - 1, image_np.shape[2] // 2, key=f"{file_type.lower()}_axial_slider")
                            fig = plot_slice(image_np[:, :, axial_slice_num], size=(3, 3), is_nifti=True)
                            st.pyplot(fig, clear_figure=True)

                        with col2:
                            coronal_slice_num = st.slider(f'{file_type} Coronal', 0, image_np.shape[1] - 1, image_np.shape[1] // 2, key=f"{file_type.lower()}_coronal_slider")
                            fig = plot_slice(image_np[:, coronal_slice_num, :], size=(3, 3), is_nifti=True)
                            st.pyplot(fig, clear_figure=True)

                        with col3:
                            sagittal_slice_num = st.slider(f'{file_type} Sagittal', 0, image_np.shape[0] - 1, image_np.shape[0] // 2, key=f"{file_type.lower()}_sagittal_slider")
                            fig = plot_slice(image_np[sagittal_slice_num, :, :], size=(3, 3), is_nifti=True)
                            st.pyplot(fig, clear_figure=True)
                    else:
                        st.warning("File could not be loaded properly. Please check the file format.")
                        
                except Exception as e:
                    st.error(f"Error processing {file_type} file: {str(e)}")
                    st.info("Please ensure the file is a valid NIfTI (.nii or .nii.gz) file.")
        
        elif processing_mode == 'legacy':
            # Original legacy mode for backward compatibility
            with tempfile.TemporaryDirectory() as temp_dir:
                is_nifti = False
                for uploaded_file in files_to_process:
                    bytes_data = uploaded_file.read()
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, 'wb') as f:
                        f.write(bytes_data)
                    if uploaded_file.name.endswith(('.nii', '.nii.gz')):
                        is_nifti = True
                
                if is_nifti:
                    image_np = load_nifti_file(file_path, "nifti_image_data")
                else:
                    image_np = load_and_store_dicom_series(temp_dir, "dicom_image_data")

            col1, col2, col3 = st.columns(3)

            # Display labels for each view with unique colors
            col1.markdown("<div class='view-label-axial'>Axial View</div>", unsafe_allow_html=True)
            col2.markdown("<div class='view-label-coronal'>Coronal View</div>", unsafe_allow_html=True)
            col3.markdown("<div class='view-label-sagittal'>Sagittal View</div>", unsafe_allow_html=True)
            
            if is_nifti:
                with col1:
                    axial_slice_num = st.slider(' ', 0, image_np.shape[2] - 1, 0, key="axial_slider")
                    fig = plot_slice(image_np[:, :, axial_slice_num], size=(3, 3), is_nifti=is_nifti)
                    st.pyplot(fig, clear_figure=True)

                with col2:
                    coronal_slice_num = st.slider('  ', 0, image_np.shape[1] - 1, 0, key="coronal_slider")
                    fig = plot_slice(image_np[:, coronal_slice_num, :], size=(3, 3), is_nifti=is_nifti)
                    st.pyplot(fig, clear_figure=True)

                with col3:
                    sagittal_slice_num = st.slider('   ', 0, image_np.shape[2] - 1, 0, key="sagittal_slider")
                    fig = plot_slice(image_np[:, :, sagittal_slice_num], size=(3, 3), is_nifti=is_nifti)
                    st.pyplot(fig, clear_figure=True)

        # PyALFE Analysis Section
        st.markdown("---")
        st.subheader("PyALFE Analysis")
        
        if st.session_state.pyalfe_installed:
            if processing_mode == 'structured':
                if st.button("Run PyALFE Analysis", key="pyalfe_button"):
                    with st.spinner("Running PyALFE analysis... This may take a few minutes."):
                        result = process_uploaded_files_with_pyalfe(flair_file, t1_file)
                        
                        if result['success']:
                            st.success("PyALFE analysis completed successfully!")
                            
                            # Display output
                            with st.expander("View Analysis Output"):
                                st.text(result['output'])
                            
                            # Display any generated files
                            if 'output_files' in result and result['output_files']:
                                st.subheader("Generated Files")
                                for file_path in result['output_files']:
                                    file_name = os.path.basename(file_path)
                                    st.text(f"📄 {file_name}")
                                    
                                    # If it's a text file or log file, offer to display contents
                                    if file_path.endswith(('.txt', '.log', '.csv')):
                                        if st.button(f"View {file_name}", key=f"view_{file_name}"):
                                            try:
                                                with open(file_path, 'r') as f:
                                                    content = f.read()
                                                st.text_area(f"Contents of {file_name}", content, height=200)
                                            except Exception as e:
                                                st.error(f"Could not read {file_name}: {str(e)}")
                        else:
                            st.error(f"Analysis failed: {result['error']}")
                            if result['output']:
                                with st.expander("Error Details"):
                                    st.text(result['output'])
                            if result.get('stderr'):
                                with st.expander("Error Details (stderr)"):
                                    st.text(result['stderr'])
            elif processing_mode == 'viewer_only':
                st.info("Please upload both FLAIR and T1 files to run PyALFE analysis.")
            else:  # legacy mode
                st.info("Legacy mode: PyALFE analysis requires structured FLAIR and T1 uploads.")
        else:
            st.info("Install PyALFE to enable automated analysis.")

if __name__ == "__main__":
    main()
                    