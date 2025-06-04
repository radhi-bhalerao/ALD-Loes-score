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
import threading
import queue
import time

def install_pyalfe():
    """
    Install PyALFE if not already installed.
    Returns True if successful, False otherwise.
    """
    try:
        # Check if pyalfe is already installed
        import pyalfe
        # Verify models are downloaded
        if verify_pyalfe_models():
            return True
        else:
            st.info("PyALFE is installed but models are missing. Downloading models...")
            return download_pyalfe_models()
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
        
        os.chdir(original_cwd)
        
        # Download and verify models
        if download_pyalfe_models():
            st.success("PyALFE installed successfully with models!")
            return True
        else:
            st.warning("PyALFE installed but model download failed. You may need to download models manually.")
            return False
        
    except Exception as e:
        os.chdir(original_cwd)
        st.error(f"Failed to install PyALFE: {str(e)}")
        return False

def verify_pyalfe_models():
    """
    Verify that PyALFE models are properly downloaded.
    """
    import os
    from pathlib import Path
    
    # Check common model locations
    cache_dir = Path.home() / ".cache" / "pyalfe"
    model_paths_to_check = [
        cache_dir / "nnunetv2" / "Dataset502_SS" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "dataset.json",
        cache_dir / "nnunetv2" / "Dataset502_SS" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "plans.json",
    ]
    
    for path in model_paths_to_check:
        if not path.exists():
            return False
    
    return True

def download_pyalfe_models():
    """
    Download PyALFE models with better error handling.
    """
    try:
        st.info("Downloading PyALFE models... This may take several minutes.")
        
        # Try multiple approaches to download models
        approaches = [
            # Approach 1: Standard download
            [sys.executable, "-c", "import pyalfe; pyalfe.download('models')"],
            # Approach 2: Force re-download
            [sys.executable, "-c", "import pyalfe; pyalfe.download('models', overwrite=True)"],
        ]
        
        for i, cmd in enumerate(approaches, 1):
            try:
                st.info(f"Trying download approach {i}...")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
                
                # Verify download was successful
                if verify_pyalfe_models():
                    st.success("Models downloaded and verified successfully!")
                    return True
                else:
                    st.warning(f"Approach {i} completed but models not verified. Trying next approach...")
                    
            except subprocess.TimeoutExpired:
                st.warning(f"Approach {i} timed out. Trying next approach...")
                continue
            except Exception as e:
                st.warning(f"Approach {i} failed: {str(e)}. Trying next approach...")
                continue
        
        # If all approaches failed, provide manual instructions
        st.error("Automatic model download failed. Please try manual download:")
        st.code("""
# In your terminal or notebook:
import pyalfe
pyalfe.download('models')

# Or force re-download:
pyalfe.download('models', overwrite=True)
        """)
        
        return False
        
    except Exception as e:
        st.error(f"Failed to download models: {str(e)}")
        return False

def debug_pyalfe_environment():
    """
    Debug function to check PyALFE environment and model status.
    """
    try:
        import pyalfe
        st.success("✅ PyALFE imported successfully")
        
        # Check model cache directory
        cache_dir = Path.home() / ".cache" / "pyalfe"
        st.info(f"Cache directory: {cache_dir}")
        st.info(f"Cache directory exists: {cache_dir.exists()}")
        
        if cache_dir.exists():
            # List contents of cache directory
            st.info("Cache directory contents:")
            for item in cache_dir.rglob("*"):
                if item.is_file():
                    st.text(f"  📄 {item.relative_to(cache_dir)}")
                elif item.is_dir():
                    st.text(f"  📁 {item.relative_to(cache_dir)}/")
        
        # Check specific model files
        model_files = [
            cache_dir / "nnunetv2" / "Dataset502_SS" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "dataset.json",
            cache_dir / "nnunetv2" / "Dataset502_SS" / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "plans.json",
        ]
        
        st.subheader("Model File Status:")
        for model_file in model_files:
            status = "✅ Found" if model_file.exists() else "❌ Missing"
            st.text(f"{status}: {model_file}")
            
    except ImportError:
        st.error("❌ PyALFE not imported - not installed")
    except Exception as e:
        st.error(f"Error checking PyALFE environment: {str(e)}")

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
    
    # Save FLAIR file with consistent naming
    if flair_file:
        # Use a consistent filename that PyALFE expects
        flair_path = os.path.join(flair_dir, "FLAIR.nii.gz")
        with open(flair_path, 'wb') as f:
            f.write(flair_file.getvalue())
        
        # Also create a backup with alternative naming if needed
        flair_alt_path = os.path.join(flair_dir, f"{accession_name}_FLAIR.nii.gz")
        with open(flair_alt_path, 'wb') as f:
            f.write(flair_file.getvalue())
    
    # Save T1 file with consistent naming
    if t1_file:
        # Use a consistent filename that PyALFE expects
        t1_path = os.path.join(t1_dir, "T1.nii.gz")
        with open(t1_path, 'wb') as f:
            f.write(t1_file.getvalue())
            
        # Also create a backup with alternative naming if needed
        t1_alt_path = os.path.join(t1_dir, f"{accession_name}_T1.nii.gz")
        with open(t1_alt_path, 'wb') as f:
            f.write(t1_file.getvalue())
    
    return accession_dir

def run_pyalfe_on_directory_realtime(accession_dir, temp_base_dir):
    """
    Run PyALFE on a directory structure with real-time output streaming and no timeout.
    
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
    
    # Debug: Print directory structure before processing
    st.write("**📁 Directory Structure:**")
    for root, dirs, files in os.walk(accession_dir):
        level = root.replace(accession_dir, '').count(os.sep)
        indent = '&nbsp;' * 4 * level
        st.markdown(f"{indent}📁 **{os.path.basename(root)}/**", unsafe_allow_html=True)
        subindent = '&nbsp;' * 4 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            st.markdown(f"{subindent}📄 {file} ({file_size:,} bytes)", unsafe_allow_html=True)
    
    try:
        # Create output directory
        output_dir = os.path.join(temp_base_dir, "processed_ald")
        os.makedirs(output_dir, exist_ok=True)
        
        # Update the existing config file with temporary paths
        config_path = "/workspaces/ALD-Loes-score/pyalfe/config_ald.ini"
        input_dir = os.path.dirname(accession_dir)  # Parent directory containing the accession
        
        # Store original config values to restore later
        original_config = None
        original_input_dir = None
        original_output_dir = None
        
        try:
            original_config = configparser.ConfigParser()
            original_config.read(config_path)
            original_input_dir = original_config.get('options', 'input_dir', fallback=None)
            original_output_dir = original_config.get('options', 'output_dir', fallback=None)
        except Exception as config_error:
            st.warning(f"Could not read original config: {config_error}")
        
        # Update config with temporary paths
        update_existing_config(input_dir, output_dir, config_path)
        st.success(f"✅ Config updated - Input: `{input_dir}`, Output: `{output_dir}`")
        
        # Get the accession name (last part of the path)
        accession_name = os.path.basename(accession_dir)
        
        # Change to the parent directory of the accession
        original_cwd = os.getcwd()
        parent_dir = os.path.dirname(accession_dir)
        os.chdir(parent_dir)
        
        # Prepare the command
        cmd = ["pyalfe", "run", "-c", config_path, accession_name]
        st.info(f"🚀 **Running Command:** `{' '.join(cmd)}`")
        st.info(f"📂 **Working Directory:** `{parent_dir}`")
        
        # Create a placeholder for real-time output
        output_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Initialize output storage
        all_stdout = []
        all_stderr = []
        
        try:
            # Start the subprocess with real-time output capture
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=parent_dir
            )
            
            # Create a queue for output
            output_queue = queue.Queue()
            
            # Start threads to capture stdout and stderr
            stdout_thread = threading.Thread(
                target=lambda: [output_queue.put(('stdout', line.strip())) 
                               for line in iter(process.stdout.readline, '') if line],
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=lambda: [output_queue.put(('stderr', line.strip())) 
                               for line in iter(process.stderr.readline, '') if line],
                daemon=True
            )
            
            stdout_thread.start()
            stderr_thread.start()
            
            # Display output in real-time
            output_lines = []
            start_time = time.time()
            
            while process.poll() is None:
                try:
                    # Check for new output
                    while True:
                        try:
                            stream_type, line = output_queue.get_nowait()
                            if stream_type == 'stdout':
                                all_stdout.append(line)
                                output_lines.append(f"📤 {line}")
                            elif stream_type == 'stderr':
                                all_stderr.append(line)
                                output_lines.append(f"⚠️ {line}")
                            
                            # Update the display with latest output (keep last 20 lines visible)
                            visible_lines = output_lines[-20:] if len(output_lines) > 20 else output_lines
                            output_text = "\n".join(visible_lines)
                            
                            with output_placeholder.container():
                                st.text_area(
                                    "🔄 **Live PyALFE Output**", 
                                    output_text, 
                                    height=300, 
                                    key=f"live_output_{len(output_lines)}"
                                )
                            
                        except queue.Empty:
                            break
                    
                    # Update progress indicator
                    elapsed = time.time() - start_time
                    progress_placeholder.info(f"⏱️ **Processing time:** {elapsed:.1f}s - **Status:** Running...")
                    
                    # Small delay to prevent excessive updates
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    st.warning("🛑 Process interrupted by user")
                    process.terminate()
                    break
            
            # Wait for process to complete and capture any remaining output
            process.wait()
            
            # Capture any remaining output
            while True:
                try:
                    stream_type, line = output_queue.get_nowait()
                    if stream_type == 'stdout':
                        all_stdout.append(line)
                        output_lines.append(f"📤 {line}")
                    elif stream_type == 'stderr':
                        all_stderr.append(line)
                        output_lines.append(f"⚠️ {line}")
                except queue.Empty:
                    break
            
            # Final output display
            final_output_text = "\n".join(output_lines)
            with output_placeholder.container():
                st.text_area(
                    "✅ **Final PyALFE Output**", 
                    final_output_text, 
                    height=400, 
                    key="final_output"
                )
            
            total_time = time.time() - start_time
            progress_placeholder.success(f"🎉 **Completed in:** {total_time:.1f}s")
            
            # Restore original config values if they existed
            if original_input_dir and original_output_dir:
                try:
                    update_existing_config(original_input_dir, original_output_dir, config_path)
                except Exception as restore_error:
                    st.warning(f"Could not restore original config: {restore_error}")
            
            if process.returncode == 0:
                # Check if output files were created
                output_files = []
                if os.path.exists(output_dir):
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            output_files.append(os.path.join(root, file))
                
                st.success(f"🎯 **PyALFE completed successfully!** Found {len(output_files)} output files.")
                
                return {
                    "success": True,
                    "output": "\n".join(all_stdout),
                    "error": None,
                    "stderr": "\n".join(all_stderr) if all_stderr else None,
                    "output_files": output_files,
                    "output_dir": output_dir,
                    "processing_time": total_time
                }
            else:
                st.error(f"❌ **PyALFE failed with return code:** {process.returncode}")
                return {
                    "success": False,
                    "error": f"PyALFE command failed with return code {process.returncode}",
                    "output": "\n".join(all_stdout),
                    "stderr": "\n".join(all_stderr),
                    "processing_time": total_time
                }
                
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        os.chdir(original_cwd)
        # Try to restore original config on error
        if original_input_dir and original_output_dir:
            try:
                update_existing_config(original_input_dir, original_output_dir, config_path)
            except:
                pass
        
        st.error(f"💥 **Unexpected error:** {str(e)}")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "output": None
        }

def process_uploaded_files_with_pyalfe_realtime(flair_file, t1_file):
    """
    Process uploaded FLAIR and T1 files with PyALFE using real-time output display.
    
    Args:
        flair_file: Streamlit uploaded FLAIR file
        t1_file: Streamlit uploaded T1 file
    
    Returns:
        dict: Results from PyALFE processing
    """
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            st.info("🔧 **Setting up PyALFE directory structure...**")
            
            # Create the PyALFE directory structure
            accession_dir = create_pyalfe_directory_structure(flair_file, t1_file, temp_dir)
            
            st.success("📁 **Directory structure created successfully!**")
            
            # Run PyALFE on the directory with real-time output
            result = run_pyalfe_on_directory_realtime(accession_dir, temp_dir)
            return result
        
        except Exception as e:
            st.error(f"💥 **Failed to process files:** {str(e)}")
            return {
                "success": False,
                "error": f"Failed to process files: {str(e)}",
                "output": None
            }

def main():
    st.set_page_config(page_title='ALD Loes Score Calculator', layout="wide")
    st.set_option('server.maxUploadSize', 1000)
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
            type=[], 
            key="flair_uploader",
            help="Upload your FLAIR (Fluid Attenuated Inversion Recovery) NIfTI file"
        )
    
    with col_upload2:
        st.markdown("**T1 Image**")
        t1_file = st.file_uploader(
            "Choose T1 NIfTI file", 
            type=[], 
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
        st.subheader("🧠 PyALFE Analysis")
        
        if st.session_state.pyalfe_installed:
            if processing_mode == 'structured':
                # Add a warning about processing time
                st.info("⏱️ **Note:** PyALFE analysis can take 5-30 minutes depending on your system. The process will show live output below.")
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    run_analysis = st.button("🚀 Run PyALFE Analysis", key="pyalfe_button")
                
                with col2:
                    if st.button("🔍 Debug PyALFE Environment", key="debug_pyalfe"):
                        with st.expander("🔧 PyALFE Environment Debug", expanded=True):
                            debug_pyalfe_environment()
                
                if run_analysis:
                    st.markdown("### 🔄 **PyALFE Processing**")
                    
                    # Run with real-time output
                    result = process_uploaded_files_with_pyalfe_realtime(flair_file, t1_file)
                    
                    if result['success']:
                        st.balloons()  # Celebration for successful completion!
                        
                        # Display summary
                        st.markdown("### 🎉 **Analysis Complete!**")
                        processing_time = result.get('processing_time', 0)
                        st.success(f"✅ **Total processing time:** {processing_time:.1f} seconds")
                        
                        # Display any generated files
                        if 'output_files' in result and result['output_files']:
                            st.markdown("### 📊 **Generated Files**")
                            
                            files_by_type = {}
                            for file_path in result['output_files']:
                                file_name = os.path.basename(file_path)
                                file_ext = os.path.splitext(file_name)[1].lower()
                                
                                if file_ext not in files_by_type:
                                    files_by_type[file_ext] = []
                                files_by_type[file_ext].append((file_name, file_path))
                            
                            # Display files organized by type
                            for ext, files in files_by_type.items():
                                st.markdown(f"**{ext.upper() or 'No Extension'} Files:**")
                                for file_name, file_path in files:
                                    col1, col2, col3 = st.columns([3, 1, 1])
                                    
                                    with col1:
                                        file_size = os.path.getsize(file_path)
                                        st.text(f"📄 {file_name} ({file_size:,} bytes)")
                                    
                                    with col2:
                                        # If it's a text file, offer to display contents
                                        if file_path.endswith(('.txt', '.log', '.csv', '.json')):
                                            if st.button(f"👁️ View", key=f"view_{file_name}"):
                                                try:
                                                    with open(file_path, 'r') as f:
                                                        content = f.read()
                                                    with st.expander(f"📖 Contents of {file_name}", expanded=True):
                                                        if file_path.endswith('.json'):
                                                            st.json(content)
                                                        else:
                                                            st.text_area(f"Contents", content, height=300, key=f"content_{file_name}")
                                                except Exception as e:
                                                    st.error(f"Could not read {file_name}: {str(e)}")
                                    
                                    with col3:
                                        # Offer download for all files
                                        try:
                                            with open(file_path, 'rb') as f:
                                                file_bytes = f.read()
                                            st.download_button(
                                                label="💾 Download",
                                                data=file_bytes,
                                                file_name=file_name,
                                                key=f"download_{file_name}"
                                            )
                                        except Exception as e:
                                            st.error(f"Download error: {str(e)}")
                        
                        # Show final output summary in an expander
                        with st.expander("📋 **Detailed Output Log**", expanded=False):
                            if result['output']:
                                st.text_area("Standard Output", result['output'], height=200)
                            if result.get('stderr'):
                                st.text_area("Error Output", result['stderr'], height=100)
                                
                    else:
                        st.error("❌ **Analysis Failed**")
                        st.error(f"**Error:** {result['error']}")
                        
                        if result.get('output'):
                            with st.expander("📋 **Output Details**", expanded=True):
                                st.text_area("Standard Output", result['output'], height=200)
                                
                        if result.get('stderr'):
                            with st.expander("⚠️ **Error Details**", expanded=True):
                                st.text_area("Error Output", result['stderr'], height=200)
                        
                        # Provide troubleshooting suggestions
                        with st.expander("🔧 **Troubleshooting Tips**", expanded=False):
                            st.markdown("""
                            **Common issues and solutions:**
                            
                            1. **Model files missing:** Click "Debug PyALFE Environment" to check model status
                            2. **File format issues:** Ensure files are valid NIfTI (.nii or .nii.gz) format
                            3. **Memory issues:** Try processing smaller images or restart the application
                            4. **Configuration issues:** Check that config file paths are correct
                            
                            **If problems persist:**
                            - Try running the debug function above
                            - Check the error output for specific error messages
                            - Verify your input files are not corrupted
                            """)
                            
            elif processing_mode == 'viewer_only':
                st.info("📝 **Please upload both FLAIR and T1 files to run PyALFE analysis.**")
                st.markdown("You currently have only one file uploaded. PyALFE requires both FLAIR and T1 images for analysis.")
                
            else:  # legacy mode
                st.info("📝 **Legacy mode detected.** PyALFE analysis requires structured FLAIR and T1 uploads using the dedicated file uploaders above.")
                
        else:
            st.warning("⚠️ **PyALFE not installed.**")
            st.markdown("Install PyALFE using the button at the top of the page to enable automated ALD Loes score analysis.")

if __name__ == "__main__":
    main()