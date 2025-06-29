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
import gc
import psutil
import shutil
import json
from scipy import stats
import pandas as pd

def keep_codespace_alive():
    """Keep codespace active during long processing"""
    def heartbeat():
        while True:
            try:
                # Simple file touch to show activity
                with open('/tmp/heartbeat.txt', 'w') as f:
                    f.write(str(time.time()))
                time.sleep(30)  # Every 30 seconds
            except:
                break
    
    # Start heartbeat in background thread
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    return thread

def create_permanent_output_directory():
    """Create a permanent output directory for PyALFE results"""
    
    # Choose appropriate base directory based on environment
    if os.getenv('CODESPACES'):
        # In GitHub Codespace
        base_output_dir = "/pyalfe_outputs"
    else:
        # Local environment - use current working directory
        base_output_dir = os.path.join(os.getcwd(), "pyalfe_outputs")
    
    try:
        os.makedirs(base_output_dir, exist_ok=True)
    except PermissionError:
        # Fallback to temp directory if permission denied
        base_output_dir = os.path.join(tempfile.gettempdir(), "pyalfe_outputs")
        os.makedirs(base_output_dir, exist_ok=True)
    
    # Create session-specific directory with timestamp
    session_timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_output_dir, f"session_{session_timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    return session_dir

def save_pyalfe_outputs(temp_output_dir, session_dir, accession_dir):
    """Save PyALFE outputs to permanent location - DEPRECATED: Now using direct permanent storage"""
    # This function is no longer needed since we're using direct permanent storage
    # Keeping for backward compatibility but will always return True
    return True

def get_folder_size(folder_path):
    """Get total size of a folder in MB"""
    total_size = 0
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except:
        pass
    return total_size / (1024 * 1024)  # Convert to MB

def show_saved_sessions():
    """Show all saved sessions with download options"""
    st.subheader("💾 Saved Sessions")
    
    base_output_dir = "/pyalfe_outputs"
    
    if not os.path.exists(base_output_dir):
        st.info("No saved sessions found.")
        return
    
    sessions = []
    for session_folder in os.listdir(base_output_dir):
        session_path = os.path.join(base_output_dir, session_folder)
        if os.path.isdir(session_path):
            session_info = {
                'folder': session_folder,
                'path': session_path,
                'size': get_folder_size(session_path),
                'modified': os.path.getmtime(session_path),
                'status': 'Unknown'
            }
            
            # Try to load session info
            info_file = os.path.join(session_path, 'session_info.json')
            if os.path.exists(info_file):
                try:
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                        session_info['status'] = info.get('status', 'Unknown')
                        session_info['timestamp'] = info.get('human_time', 'Unknown')
                except:
                    pass
            
            sessions.append(session_info)
    
    # Sort by modification time
    sessions.sort(key=lambda x: x['modified'], reverse=True)
    
    for session in sessions:
        with st.expander(f"📁 **{session['folder']}** - {session['status']} ({session['size']:.1f} MB)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Status:** {session['status']}")
                st.write(f"**Size:** {session['size']:.1f} MB")
                if 'timestamp' in session:
                    st.write(f"**Time:** {session['timestamp']}")
            
            with col2:
                # List files in session
                session_files = []
                for root, dirs, files in os.walk(session['path']):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), session['path'])
                        session_files.append(rel_path)
                
                st.write(f"**Files:** {len(session_files)}")
                if st.button(f"📋 List Files", key=f"list_{session['folder']}"):
                    st.write("**File List:**")
                    for file in sorted(session_files)[:20]:  # Show first 20
                        st.text(f"  📄 {file}")
                    if len(session_files) > 20:
                        st.text(f"  ... and {len(session_files) - 20} more files")
            
            with col3:
                # Download individual files
                final_outputs_dir = os.path.join(session['path'], 'final_outputs')
                if os.path.exists(final_outputs_dir):
                    for root, dirs, files in os.walk(final_outputs_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'rb') as f:
                                    file_bytes = f.read()
                                st.download_button(
                                    label=f"💾 {file}",
                                    data=file_bytes,
                                    file_name=file,
                                    key=f"download_{session['folder']}_{file}"
                                )
                            except Exception as e:
                                st.error(f"Download error for {file}: {e}")

def install_pyalfe():
    """Install PyALFE if not already installed."""
    try:
        import pyalfe
        if verify_pyalfe_models():
            return True
        else:
            st.info("PyALFE is installed but models are missing. Downloading models...")
            return download_pyalfe_models()
    except ImportError:
        pass
    
    pyalfe_dir = os.path.join(os.getcwd(), 'pyalfe')
    if not os.path.exists(pyalfe_dir):
        st.error(f"PyALFE directory not found at: {pyalfe_dir}")
        st.info("Please clone PyALFE: `git clone https://github.com/reghbali/pyalfe.git`")
        return False
    
    try:
        st.info("Installing PyALFE... This may take a few minutes.")
        
        original_cwd = os.getcwd()
        os.chdir(pyalfe_dir)
        
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "build"], 
                      check=True, capture_output=True)
        subprocess.run([sys.executable, "-m", "build"], 
                      check=True, capture_output=True)
        
        dist_dir = os.path.join(pyalfe_dir, "dist")
        wheel_files = [f for f in os.listdir(dist_dir) if f.endswith('.whl')]
        if not wheel_files:
            raise Exception("No wheel file found after build")
        
        wheel_file = os.path.join(dist_dir, wheel_files[0])
        subprocess.run([sys.executable, "-m", "pip", "install", wheel_file], 
                      check=True, capture_output=True)
        
        os.chdir(original_cwd)
        
        if download_pyalfe_models():
            st.success("PyALFE installed successfully with models!")
            return True
        else:
            st.warning("PyALFE installed but model download failed.")
            return False
        
    except Exception as e:
        os.chdir(original_cwd)
        st.error(f"Failed to install PyALFE: {str(e)}")
        return False

def verify_pyalfe_models():
    """Verify that PyALFE models are properly downloaded."""
    from pathlib import Path
    
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
    """Download PyALFE models with better error handling."""
    try:
        st.info("Downloading PyALFE models... This may take several minutes.")
        
        approaches = [
            [sys.executable, "-c", "import pyalfe; pyalfe.download('models')"],
            [sys.executable, "-c", "import pyalfe; pyalfe.download('models', overwrite=True)"],
        ]
        
        for i, cmd in enumerate(approaches, 1):
            try:
                st.info(f"Trying download approach {i}...")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
                
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
        
        st.error("Automatic model download failed.")
        return False
        
    except Exception as e:
        st.error(f"Failed to download models: {str(e)}")
        return False

def debug_pyalfe_environment():
    """Debug function to check PyALFE environment and model status."""
    try:
        import pyalfe
        st.success("✅ PyALFE imported successfully")
        
        cache_dir = Path.home() / ".cache" / "pyalfe"
        st.info(f"Cache directory: {cache_dir}")
        st.info(f"Cache directory exists: {cache_dir.exists()}")
        
        if cache_dir.exists():
            st.info("Cache directory contents:")
            for item in cache_dir.rglob("*"):
                if item.is_file():
                    st.text(f"  📄 {item.relative_to(cache_dir)}")
                elif item.is_dir():
                    st.text(f"  📁 {item.relative_to(cache_dir)}/")
        
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

def update_existing_config(input_dir, output_dir, config_path=None):
    """Update the existing config file with new input and output directories."""
    
    # Auto-detect config path based on environment
    if config_path is None:
        if os.getenv('CODESPACES'):
            config_path = "/workspaces/ALD-Loes-score/pyalfe/config_ald.ini"
        else:
            # Local environment - look in current directory
            config_path = os.path.join(os.getcwd(), "pyalfe", "config_ald.ini")
            if not os.path.exists(config_path):
                config_path = "./config_ald.ini"  # Fallback
    
    try:
        config = configparser.ConfigParser()
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
        config.read(config_path)
        
        if 'options' not in config:
            config['options'] = {}
        
        config['options']['input_dir'] = input_dir
        config['options']['output_dir'] = output_dir
        
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
    fig, ax = plt.subplots(figsize=size)
    canvas_size = max(slice.shape)
    canvas = np.full((canvas_size, canvas_size), fill_value=slice.min(), dtype=slice.dtype)
    x_offset = (canvas_size - slice.shape[0]) // 2
    y_offset = (canvas_size - slice.shape[1]) // 2
    canvas[x_offset:x_offset+slice.shape[0], y_offset:y_offset+slice.shape[1]] = slice
    fig.patch.set_facecolor('black')
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
            nifti_img = nib.load(filepath)
            image_np = np.asanyarray(nifti_img.dataobj)
            st.session_state[session_key] = image_np
        except Exception as e:
            st.error(f"Error loading NIfTI file: {str(e)}")
            
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
            
            st.session_state[session_key] = np.zeros((100, 100, 50))
            
    return st.session_state[session_key]

def create_pyalfe_directory_structure(flair_file, t1_file, output_dir, accession_name="ACCESSION"):
    """Create the directory structure expected by PyALFE in the permanent output directory."""
    # Create input directory within the permanent output directory
    input_dir = os.path.join(output_dir, "input")
    accession_dir = os.path.join(input_dir, accession_name)
    
    flair_dir = os.path.join(accession_dir, "FLAIR")
    t1_dir = os.path.join(accession_dir, "T1")
    
    os.makedirs(flair_dir, exist_ok=True)
    os.makedirs(t1_dir, exist_ok=True)
    
    if flair_file:
        flair_path = os.path.join(flair_dir, "FLAIR.nii.gz")
        with open(flair_path, 'wb') as f:
            f.write(flair_file.getvalue())
    
    if t1_file:
        t1_path = os.path.join(t1_dir, "T1.nii.gz")
        with open(t1_path, 'wb') as f:
            f.write(t1_file.getvalue())
    
    return accession_dir, input_dir

def run_pyalfe_on_directory_realtime(accession_dir, input_dir, output_dir, use_cpu_only=True, cpu_processes=2):
    """Run PyALFE on a directory structure with configurable CPU/GPU processing."""
    try:
        import pyalfe
    except ImportError:
        return {
            "success": False,
            "error": "PyALFE not installed. Please install it first.",
            "output": None
        }
    
    if not os.path.exists(accession_dir):
        return {
            "success": False,
            "error": f"Accession directory not found at: {accession_dir}",
            "output": None
        }
    
    # Debug: Print directory structure before processing
    st.write("**📁 Input Directory Structure:**")
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
        # Create PyALFE output directory within the permanent output directory
        pyalfe_output_dir = os.path.join(output_dir, "pyalfe_results")
        os.makedirs(pyalfe_output_dir, exist_ok=True)
        
        # Update the existing config file with permanent paths
        config_path = os.path.join(os.getcwd(), "pyalfe", "config_ald.ini") if not os.getenv('CODESPACES') else "/workspaces/ALD-Loes-score/pyalfe/config_ald.ini"
        
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
        
        # Update config with permanent paths
        update_existing_config(input_dir, pyalfe_output_dir, config_path)
        st.success(f"✅ Config updated - Input: `{input_dir}`, Output: `{pyalfe_output_dir}`")
        st.info(f"💾 **Results will be saved directly to:** `{output_dir}`")
        
        # Get the accession name (last part of the path)
        accession_name = os.path.basename(accession_dir)
        
        # Change to the input directory
        original_cwd = os.getcwd()
        os.chdir(input_dir)
        
        # Set up environment variables based on processing mode
        env = os.environ.copy()
        
        if use_cpu_only:
            # Force CPU-only processing
            env['CUDA_VISIBLE_DEVICES'] = ''
            env['TORCH_CUDA_DEVICE_COUNT'] = '0'
            env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            
            # CPU-specific settings
            env['nnUNet_n_proc_DA'] = str(cpu_processes)
            env['nnUNet_compile'] = 'False'
            env['OMP_NUM_THREADS'] = str(min(cpu_processes * 2, 8))
            env['MKL_NUM_THREADS'] = str(min(cpu_processes * 2, 8))
            
            processing_mode_str = f"CPU-only ({cpu_processes} processes)"
            st.info(f"🖥️ **Processing Mode:** {processing_mode_str}")
            st.info("💡 **Note:** CPU processing is slower but more reliable")
            
        else:
            # GPU processing with memory management
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            env['CUDA_LAUNCH_BLOCKING'] = '1'
            env['nnUNet_n_proc_DA'] = '1'  # Conservative for GPU
            
            processing_mode_str = "GPU with memory management"
            st.info(f"🚀 **Processing Mode:** {processing_mode_str}")
            st.warning("⚠️ **Note:** GPU processing may fail if insufficient memory")
        
        # Prepare the command
        cmd = ["pyalfe", "run", "-c", config_path, "--no-overwrite", accession_name]
        st.info(f"🚀 **Running Command:** `{' '.join(cmd)}`")
        st.info(f"📂 **Working Directory:** `{input_dir}`")
        
        # Create a placeholder for real-time output
        output_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Initialize output storage
        all_stdout = []
        all_stderr = []
        
        try:
            # Start the subprocess with real-time output capture and updated environment
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=input_dir,
                env=env  # Use the updated environment
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
                    progress_placeholder.info(f"⏱️ **Processing time:** {elapsed:.1f}s - **Status:** Running ({processing_mode_str})...")
                    
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
            progress_placeholder.success(f"🎉 **Completed in:** {total_time:.1f}s using {processing_mode_str}")
            
            # Create a session info file in the output directory
            session_info = {
                'timestamp': time.time(),
                'human_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'output_dir': output_dir,
                'pyalfe_output_dir': pyalfe_output_dir,
                'input_dir': input_dir,
                'processing_mode': processing_mode_str,
                'use_cpu_only': use_cpu_only,
                'cpu_processes': cpu_processes if use_cpu_only else None,
                'status': 'completed' if process.returncode == 0 else 'failed'
            }
            
            with open(os.path.join(output_dir, 'session_info.json'), 'w') as f:
                json.dump(session_info, f, indent=2)
            
            # Restore original config values if they existed
            if original_input_dir and original_output_dir:
                try:
                    update_existing_config(original_input_dir, original_output_dir, config_path)
                except Exception as restore_error:
                    st.warning(f"Could not restore original config: {restore_error}")
            
            if process.returncode == 0:
                # Check if output files were created
                output_files = []
                
                # Check PyALFE output files
                if os.path.exists(pyalfe_output_dir):
                    for root, dirs, files in os.walk(pyalfe_output_dir):
                        for file in files:
                            output_files.append(os.path.join(root, file))
                
                st.success(f"🎯 **PyALFE completed successfully using {processing_mode_str}!** Found {len(output_files)} output files.")
                
                return {
                    "success": True,
                    "output": "\n".join(all_stdout),
                    "error": None,
                    "stderr": "\n".join(all_stderr) if all_stderr else None,
                    "output_files": output_files,
                    "output_dir": output_dir,
                    "pyalfe_output_dir": pyalfe_output_dir,
                    "processing_time": total_time,
                    "processing_mode": processing_mode_str
                }
            else:
                st.error(f"❌ **PyALFE failed with return code:** {process.returncode}")
                return {
                    "success": False,
                    "error": f"PyALFE command failed with return code {process.returncode}",
                    "output": "\n".join(all_stdout),
                    "stderr": "\n".join(all_stderr),
                    "processing_time": total_time,
                    "processing_mode": processing_mode_str
                }
                
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        os.chdir(original_cwd)
        # Try to restore original config on error
        if 'original_input_dir' in locals() and 'original_output_dir' in locals():
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

def process_uploaded_files_with_pyalfe_realtime(flair_file, t1_file, use_cpu_only=True, cpu_processes=2):
    """Process uploaded FLAIR and T1 files with PyALFE using direct permanent storage."""
    
    # Create permanent output directory
    output_dir = create_permanent_output_directory()
    st.info(f"📁 **Output directory created:** `{output_dir}`")
    
    try:
        st.info("🔧 **Setting up PyALFE directory structure...**")
        
        # Create the PyALFE directory structure in permanent location
        accession_dir, input_dir = create_pyalfe_directory_structure(flair_file, t1_file, output_dir)
        
        st.success("📁 **Directory structure created successfully!**")
        
        # Run PyALFE with direct output to permanent folder
        result = run_pyalfe_on_directory_realtime(
            accession_dir, input_dir, output_dir, use_cpu_only=use_cpu_only, cpu_processes=cpu_processes
        )
        return result
    
    except Exception as e:
        st.error(f"💥 **Failed to process files:** {str(e)}")
        return {
            "success": False,
            "error": f"Failed to process files: {str(e)}",
            "output": None
        }

def clear_streamlit_session_cache():
    """Clear Streamlit-specific session data"""
    keys_to_clear = [
        'flair_image_data', 't1_image_data', 'nifti_image_data', 
        'dicom_image_data', 'processed_data', 'pyalfe_results'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear Streamlit cache
    st.cache_data.clear()
    st.cache_resource.clear()

def clear_all_memory():
    """Basic memory clearing function"""
    try:
        # Clear Python garbage collection
        gc.collect()
        
        # Clear matplotlib cache
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        return True
    except Exception as e:
        st.warning(f"Memory clearing had some issues: {str(e)}")
        return False

def get_memory_usage():
    """Get current memory usage"""
    memory = psutil.virtual_memory()
    return {
        'percent': memory.percent,
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'total_gb': memory.total / (1024**3)
    }

def log_memory_usage(stage=""):
    """Log memory usage with stage information"""
    mem = get_memory_usage()
    st.info(f"🖥️ **Memory {stage}:** {mem['percent']:.1f}% used | {mem['available_gb']:.1f}GB available | {mem['used_gb']:.1f}GB used")
    return mem

def load_normative_data(normative_csv_path):
    """Load and process normative brain volume data"""
    try:
        df = pd.read_csv(normative_csv_path)
        # Convert month_age from list format to numeric if needed
        if 'month_age' in df.columns:
            df['month_age'] = df['month_age'].apply(lambda x: float(str(x).strip('[]')) if isinstance(x, str) else x)
        return df
    except Exception as e:
        st.error(f"Error loading normative data: {str(e)}")
        return None

def calculate_age_specific_stats(normative_df, patient_age_months, age_window=12):
    """Calculate mean and std for age-matched controls"""
    # Find subjects within age window (±6 months by default)
    age_matched = normative_df[
        (normative_df['month_age'] >= patient_age_months - age_window/2) & 
        (normative_df['month_age'] <= patient_age_months + age_window/2)
    ]
    
    if len(age_matched) < 3:  # If too few matches, expand window
        age_window = 24
        age_matched = normative_df[
            (normative_df['month_age'] >= patient_age_months - age_window/2) & 
            (normative_df['month_age'] <= patient_age_months + age_window/2)
        ]
    
    if len(age_matched) < 3:  # If still too few, use all data
        age_matched = normative_df
        st.warning(f"Limited age-matched data. Using all normative subjects (n={len(age_matched)})")
    else:
        st.info(f"Using {len(age_matched)} age-matched controls (±{age_window/2} months)")
    
    return age_matched

def assess_lesion_presence(lesion_df, threshold=0.1):
    """Assess presence of lesions in different brain regions"""
    lesion_presence = {}
    
    # Define region mappings based on Loes scoring system
    region_mappings = {
        # Supratentorial white matter regions
        'parieto_occipital': ['lesion_volume_in_Parietal_Occipital', 'lesion_volume_in_Parietal', 'lesion_volume_in_Occipital'],
        'anterior_temporal': ['lesion_volume_in_AnteriorTemporal', 'lesion_volume_in_Temporal'],
        'frontal': ['lesion_volume_in_Frontal'],
        
        # Corpus callosum
        'corpus_callosum_genu': ['lesion_volume_in_CorpusCallosum_Genu'],
        'corpus_callosum_body': ['lesion_volume_in_CorpusCallosum_Body'],
        'corpus_callosum_splenium': ['lesion_volume_in_CorpusCallosum_Splenium'],
        
        # Visual pathway
        'optic_radiations': ['lesion_volume_in_Right OR', 'lesion_volume_in_Left OR'],
        'lateral_geniculate': ['lesion_volume_in_Right LGN', 'lesion_volume_in_Left LGN'],
        'meyers_loop': ['lesion_volume_in_Right Meyers Loop', 'lesion_volume_in_Left Meyers Loop'],
        
        # Auditory pathway
        'medial_geniculate': ['lesion_volume_in_Right Medial Geniculate', 'lesion_volume_in_Left Medial Geniculate'],
        'brachium_inf_col': ['lesion_volume_in_Right Brachium Inf Col', 'lesion_volume_in_Left Brachium Inf Col'],
        'lateral_lemniscus': ['lesion_volume_in_Right Lat Lemniscus', 'lesion_volume_in_Left Lat Lemniscus'],
        'pons': ['lesion_volume_in_Pons'],
        
        # Projection fibers
        'internal_capsule': ['lesion_volume_in_Right ALIC', 'lesion_volume_in_Left ALIC'],
        'brain_stem': ['lesion_volume_in_Brain Stem'],
        
        # Other regions
        'basal_ganglia': ['lesion_volume_in_Basal Ganglia'],
        'cerebellum': ['lesion_volume_in_Cerebellum'],
        'anterior_thalamus': ['lesion_volume_in_Right Anterior Thalamus', 'lesion_volume_in_Left Anterior Thalamus']
    }
    
    for region, columns in region_mappings.items():
        total_lesion_volume = 0
        bilateral_involvement = False
        unilateral_involvement = False
        
        # Check for bilateral vs unilateral involvement
        left_columns = [col for col in columns if 'Left' in col]
        right_columns = [col for col in columns if 'Right' in col]
        
        if left_columns and right_columns:
            left_volume = sum([lesion_df[col].iloc[0] if col in lesion_df.columns else 0 for col in left_columns])
            right_volume = sum([lesion_df[col].iloc[0] if col in lesion_df.columns else 0 for col in right_columns])
            
            if left_volume > threshold and right_volume > threshold:
                bilateral_involvement = True
            elif left_volume > threshold or right_volume > threshold:
                unilateral_involvement = True
            
            total_lesion_volume = left_volume + right_volume
        else:
            total_lesion_volume = sum([lesion_df[col].iloc[0] if col in lesion_df.columns else 0 for col in columns])
        
        lesion_presence[region] = {
            'total_volume': total_lesion_volume,
            'present': total_lesion_volume > threshold,
            'bilateral': bilateral_involvement,
            'unilateral': unilateral_involvement
        }
    
    return lesion_presence

def calculate_atrophy_scores(patient_volumes, normative_stats, z_threshold=2.0):
    """Calculate atrophy scores based on volume deviations from normal"""
    atrophy_scores = {}
    
    # Define volume mappings for atrophy assessment
    atrophy_regions = {
        'parieto_occipital': 'volume_of_Parietal_Occipital',
        'anterior_temporal': 'volume_of_AnteriorTemporal', 
        'frontal': 'volume_of_Frontal',
        'corpus_callosum_genu': 'volume_of_CorpusCallosum_Genu',
        'corpus_callosum_splenium': 'volume_of_CorpusCallosum_Splenium',
        'cerebellum': 'volume_of_Cerebellum',  # Assuming this exists in normative data
        'brain_stem': 'volume_of_brain_stem',
        'total_brain': 'total_brain_volume',
        'ventricles': 'total_ventricles_volume'
    }
    
    for region, column in atrophy_regions.items():
        if column in normative_stats.columns:
            mean_vol = normative_stats[column].mean()
            std_vol = normative_stats[column].std()
            
            # For patient volumes, we need to extract from the appropriate source
            # This would need to be adapted based on how patient volume data is structured
            patient_vol = None  # This needs to be implemented based on your data structure
            
            if patient_vol is not None:
                z_score = (patient_vol - mean_vol) / std_vol
                
                # Negative z-score indicates volume loss (atrophy)
                atrophy_scores[region] = {
                    'z_score': z_score,
                    'atrophic': z_score < -z_threshold,
                    'patient_volume': patient_vol,
                    'normal_mean': mean_vol,
                    'normal_std': std_vol
                }
    
    return atrophy_scores

def calculate_loes_score(lesion_presence, atrophy_scores, patient_age_months):
    """Calculate the complete Loes score based on lesions and atrophy"""
    
    score = 0
    score_breakdown = {}
    
    # Supratentorial white matter scoring (max 18 points)
    supratentorial_regions = ['parieto_occipital', 'anterior_temporal', 'frontal']
    
    for region in supratentorial_regions:
        region_score = 0
        
        # Periventricular (2 points)
        if lesion_presence.get(region, {}).get('present', False):
            if lesion_presence[region]['bilateral']:
                region_score += 2
            elif lesion_presence[region]['unilateral']:
                region_score += 1
        
        # Central white matter (2 points) - using same logic for now
        if lesion_presence.get(region, {}).get('present', False):
            if lesion_presence[region]['bilateral']:
                region_score += 2
            elif lesion_presence[region]['unilateral']:
                region_score += 1
        
        # Subcortical (2 points) - using same logic for now
        if lesion_presence.get(region, {}).get('present', False):
            if lesion_presence[region]['bilateral']:
                region_score += 2
            elif lesion_presence[region]['unilateral']:
                region_score += 1
        
        score += region_score
        score_breakdown[f'{region}_white_matter'] = region_score
    
    # Corpus callosum (max 6 points)
    cc_regions = ['corpus_callosum_genu', 'corpus_callosum_body', 'corpus_callosum_splenium']
    cc_score = 0
    
    for cc_region in cc_regions:
        if lesion_presence.get(cc_region, {}).get('present', False):
            cc_score += 2
    
    score += cc_score
    score_breakdown['corpus_callosum'] = cc_score
    
    # Visual pathway (max 3 points)
    visual_regions = ['optic_radiations', 'lateral_geniculate', 'meyers_loop']
    visual_score = 0
    
    for visual_region in visual_regions:
        if lesion_presence.get(visual_region, {}).get('present', False):
            visual_score += 1
    
    score += visual_score
    score_breakdown['visual_pathway'] = visual_score
    
    # Auditory pathway (max 3 points)
    auditory_regions = ['medial_geniculate', 'brachium_inf_col', 'lateral_lemniscus', 'pons']
    auditory_score = 0
    
    for auditory_region in auditory_regions:
        if lesion_presence.get(auditory_region, {}).get('present', False):
            if auditory_region == 'pons':
                auditory_score += 1  # Pons gets 1 point
            else:
                auditory_score += 0.5  # Other auditory structures get 0.5 points each
    
    score += auditory_score
    score_breakdown['auditory_pathway'] = auditory_score
    
    # Projection fibers (max 2 points)
    projection_score = 0
    if lesion_presence.get('internal_capsule', {}).get('present', False):
        projection_score += 1
    if lesion_presence.get('brain_stem', {}).get('present', False):
        projection_score += 1
    
    score += projection_score
    score_breakdown['projection_fibers'] = projection_score
    
    # Other regions (max 2 points)
    other_score = 0
    if lesion_presence.get('basal_ganglia', {}).get('present', False):
        other_score += 1
    if lesion_presence.get('cerebellum', {}).get('present', False):
        other_score += 1
    if lesion_presence.get('anterior_thalamus', {}).get('present', False):
        other_score += 1
    
    score += min(other_score, 2)  # Cap at 2 points
    score_breakdown['other_regions'] = min(other_score, 2)
    
    # Atrophy scoring (max 3 points for global atrophy)
    global_atrophy_score = 0
    
    # Calculate global atrophy based on ventricular expansion and brain volume loss
    if 'ventricles' in atrophy_scores:
        ventricle_z = atrophy_scores['ventricles']['z_score']
        if ventricle_z > 2:  # Enlarged ventricles
            global_atrophy_score += 1
        if ventricle_z > 3:
            global_atrophy_score += 1
        if ventricle_z > 4:
            global_atrophy_score += 1
    
    # Add focal atrophy points based on regional volume loss
    focal_atrophy_score = 0
    focal_regions = ['parieto_occipital', 'anterior_temporal', 'frontal', 
                    'corpus_callosum_genu', 'corpus_callosum_splenium', 
                    'cerebellum', 'brain_stem']
    
    for region in focal_regions:
        if region in atrophy_scores and atrophy_scores[region]['atrophic']:
            focal_atrophy_score += 0.5  # 0.5 points per region with focal atrophy
    
    atrophy_total = global_atrophy_score + focal_atrophy_score
    score += atrophy_total
    score_breakdown['atrophy'] = atrophy_total
    score_breakdown['global_atrophy'] = global_atrophy_score
    score_breakdown['focal_atrophy'] = focal_atrophy_score
    
    return min(score, 34), score_breakdown  # Cap at maximum score of 34

def display_loes_score_interface():
    """Display the Loes score calculation interface"""
    st.markdown("---")
    st.subheader("🧮 Automated Loes Score Calculator")
    
    # Patient age input
    col1, col2 = st.columns(2)
    
    with col1:
        patient_age_years = st.number_input(
            "Patient Age (years)", 
            min_value=0.0, 
            max_value=100.0, 
            value=10.0, 
            step=0.1,
            help="Enter the patient's age to calculate age-matched normative statistics"
        )
        patient_age_months = patient_age_years * 12
    
    with col2:
        normative_csv_path = st.text_input(
            "Path to Normative Data CSV",
            value="*/volume_measurements_with_age_PING_fixed.csv",
            help="Path to the CSV file containing normative brain volume data"
        )
    
    # File path inputs
    st.markdown("### 📁 **Data File Paths**")
    
    lesion_csv_path = st.text_input(
        "Lesion Measures CSV Path",
        value="/wynton/protected/group/rsl/radiogenomics/ald_cont/pyalfe_outputs/session_20250617_221416/pyalfe_results/ACCESSION/FLAIR/quantification/ACCESSION_IndividualLesionMeasures.csv",
        help="Path to the PyALFE output CSV containing individual lesion measurements"
    )
    
    # Advanced options
    with st.expander("⚙️ **Advanced Options**", expanded=False):
        lesion_threshold = st.slider(
            "Lesion Volume Threshold (mm³)", 
            min_value=0.0, 
            max_value=10.0, 
            value=0.1, 
            step=0.1,
            help="Minimum lesion volume to consider as 'present'"
        )
        
        atrophy_z_threshold = st.slider(
            "Atrophy Z-Score Threshold", 
            min_value=1.0, 
            max_value=4.0, 
            value=2.0, 
            step=0.1,
            help="Z-score threshold for considering volume loss as atrophy"
        )
        
        age_window = st.slider(
            "Age Matching Window (months)", 
            min_value=6, 
            max_value=36, 
            value=12, 
            step=6,
            help="Age window for selecting normative controls"
        )
    
    # Calculate button
    if st.button("🧮 Calculate Loes Score", key="calculate_loes"):
        
        # Validate file paths
        if not os.path.exists(lesion_csv_path):
            st.error(f"❌ Lesion CSV file not found: {lesion_csv_path}")
            return
        
        if not os.path.exists(normative_csv_path):
            st.error(f"❌ Normative data CSV file not found: {normative_csv_path}")
            return
        
        try:
            # Load data
            st.info("📊 Loading data...")
            
            lesion_df = pd.read_csv(lesion_csv_path)
            normative_df = load_normative_data(normative_csv_path)
            
            if normative_df is None:
                return
            
            st.success(f"✅ Loaded lesion data: {lesion_df.shape[0]} rows, {lesion_df.shape[1]} columns")
            st.success(f"✅ Loaded normative data: {normative_df.shape[0]} subjects")
            
            # Calculate age-matched statistics
            st.info("📈 Calculating age-matched normative statistics...")
            age_matched_controls = calculate_age_specific_stats(
                normative_df, patient_age_months, age_window
            )
            
            # Assess lesion presence
            st.info("🔍 Assessing lesion presence...")
            lesion_presence = assess_lesion_presence(lesion_df, lesion_threshold)
            
            # Calculate atrophy scores (placeholder - needs patient volume data)
            st.info("📏 Calculating atrophy scores...")
            atrophy_scores = calculate_atrophy_scores(
                {}, age_matched_controls, atrophy_z_threshold
            )
            
            # Calculate final Loes score
            st.info("🧮 Calculating Loes score...")
            loes_score, score_breakdown = calculate_loes_score(
                lesion_presence, atrophy_scores, patient_age_months
            )
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 **Loes Score Results**")
            
            # Main score display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="**Total Loes Score**",
                    value=f"{loes_score:.1f} / 34",
                    help="Complete Loes severity score"
                )
            
            with col2:
                severity = "Mild" if loes_score < 9 else "Moderate" if loes_score < 20 else "Severe"
                color = "🟢" if loes_score < 10 else "🟡" if loes_score < 20 else "🔴"
                st.metric(
                    label="**Severity**",
                    value=f"{color} {severity}",
                    help="Severity classification based on score"
                )
            
            with col3:
                st.metric(
                    label="**Patient Age**",
                    value=f"{patient_age_years:.1f} years",
                    help="Patient age used for normative matching"
                )
            
            # Score breakdown
            st.markdown("### 📊 **Score Breakdown**")
            
            breakdown_df = pd.DataFrame([
                {"Component": "Supratentorial White Matter", "Score": score_breakdown.get('parieto_occipital_white_matter', 0) + score_breakdown.get('anterior_temporal_white_matter', 0) + score_breakdown.get('frontal_white_matter', 0), "Max": 18},
                {"Component": "Corpus Callosum", "Score": score_breakdown.get('corpus_callosum', 0), "Max": 6},
                {"Component": "Visual Pathway", "Score": score_breakdown.get('visual_pathway', 0), "Max": 3},
                {"Component": "Auditory Pathway", "Score": score_breakdown.get('auditory_pathway', 0), "Max": 3},
                {"Component": "Projection Fibers", "Score": score_breakdown.get('projection_fibers', 0), "Max": 2},
                {"Component": "Other Regions", "Score": score_breakdown.get('other_regions', 0), "Max": 2},
                {"Component": "Global Atrophy", "Score": score_breakdown.get('global_atrophy', 0), "Max": 3},
                {"Component": "Focal Atrophy", "Score": score_breakdown.get('focal_atrophy', 0), "Max": "Variable"}
            ])
            
            st.dataframe(breakdown_df, use_container_width=True)
            
            # Lesion presence summary
            st.markdown("### 🎯 **Lesion Presence Summary**")
            
            lesion_summary = []
            for region, data in lesion_presence.items():
                if data['present']:
                    involvement = "Bilateral" if data['bilateral'] else "Unilateral" if data['unilateral'] else "Present"
                    lesion_summary.append({
                        "Region": region.replace('_', ' ').title(),
                        "Status": f"✅ {involvement}",
                        "Volume (mm³)": f"{data['total_volume']:.2f}"
                    })
                else:
                    lesion_summary.append({
                        "Region": region.replace('_', ' ').title(),
                        "Status": "❌ Not Involved",
                        "Volume (mm³)": "0.00"
                    })
            
            lesion_df_display = pd.DataFrame(lesion_summary)
            st.dataframe(lesion_df_display, use_container_width=True)
            
            # Export options
            st.markdown("### 💾 **Export Results**")
            
            # Prepare export data
            export_data = {
                'patient_age_years': patient_age_years,
                'patient_age_months': patient_age_months,
                'total_loes_score': loes_score,
                'severity': severity,
                'score_breakdown': score_breakdown,
                'lesion_presence': lesion_presence,
                'calculation_timestamp': pd.Timestamp.now().isoformat()
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Copy Score to Clipboard"):
                    st.code(f"Loes Score: {loes_score:.1f}/34 ({severity})")
            
            with col2:
                export_json = pd.Series(export_data).to_json(indent=2)
                st.download_button(
                    label="💾 Download Full Report (JSON)",
                    data=export_json,
                    file_name=f"loes_score_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        except Exception as e:
            st.error(f"❌ **Error calculating Loes score:** {str(e)}")
            st.exception(e)

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

    # Sidebar for session management and memory
    st.sidebar.markdown("### 💾 Session Management")
    
    if st.sidebar.button("📂 Show Saved Sessions"):
        show_saved_sessions()
    
    # Show current storage usage
    base_output_dir = "/pyalfe_outputs"
    if os.path.exists(base_output_dir):
        total_size = get_folder_size(base_output_dir)
        st.sidebar.info(f"💽 **Storage used:** {total_size:.1f} MB")
        
        if total_size > 1000:  # Over 1GB
            st.sidebar.warning("⚠️ Consider cleaning old sessions")

    st.sidebar.markdown("### 🧹 Memory Management")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🧹 Clear Memory", key="clear_memory"):
            clear_streamlit_session_cache()
            clear_all_memory()
            st.success("Memory cleared!")
            st.rerun()
    
    with col2:
        if st.button("📊 Check Memory", key="check_memory"):
            log_memory_usage("current")
    
    # Show current memory usage
    mem = get_memory_usage()
    if mem['percent'] > 75:
        st.sidebar.warning(f"⚠️ Memory: {mem['percent']:.1f}% used")
    else:
        st.sidebar.info(f"🖥️ Memory: {mem['percent']:.1f}% used")

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
                # Add processing mode selection
                st.markdown("### ⚙️ **Processing Configuration**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    processing_device = st.radio(
                        "**Select Processing Device:**",
                        ["CPU Only (Recommended)", "GPU (if available)"],
                        index=0,  # Default to CPU
                        help="CPU processing is more stable and avoids memory issues, but slower than GPU"
                    )
                
                with col2:
                    if processing_device == "CPU Only (Recommended)":
                        st.info("🖥️ **CPU Processing**: Stable, reliable, works on all systems")
                        cpu_processes = st.slider("CPU Processes", 1, 8, 2, help="Number of CPU processes for data augmentation")
                    else:
                        st.warning("⚠️ **GPU Processing**: Faster but may cause memory errors")
                        st.text("Will attempt GPU processing with memory management")
                
                # Add a warning about processing time
                st.info("⏱️ **Note:** PyALFE analysis can take 5-30 minutes depending on your system and processing mode.")
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    run_analysis = st.button("🚀 Run PyALFE Analysis", key="pyalfe_button")
                
                with col2:
                    if st.button("🔍 Debug PyALFE Environment", key="debug_pyalfe"):
                        with st.expander("🔧 PyALFE Environment Debug", expanded=True):
                            debug_pyalfe_environment()
                
                if run_analysis:
                    st.info("🔄 **Starting analysis... (keeping connection alive)**")
    
                    # Keep codespace active
                    heartbeat_thread = keep_codespace_alive()
    
                    # Pass processing configuration to the analysis function
                    use_cpu_only = (processing_device == "CPU Only (Recommended)")
                    cpu_proc_count = cpu_processes if use_cpu_only else 2
    
                    st.markdown("### 🔄 **PyALFE Processing**")
                    result = process_uploaded_files_with_pyalfe_realtime(
                        flair_file, t1_file, use_cpu_only=use_cpu_only, cpu_processes=cpu_proc_count
                    )
                    
                    if result['success']:
                        st.balloons()  # Celebration for successful completion!
                        
                        # Display summary
                        st.markdown("### 🎉 **Analysis Complete!**")
                        processing_time = result.get('processing_time', 0)
                        st.success(f"✅ **Total processing time:** {processing_time:.1f} seconds")
                        
                        # Show output directory
                        if 'output_dir' in result:
                            st.info(f"📁 **All outputs saved to:** `{result['output_dir']}`")
                        
                        # Display any generated files from output directory
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
                                        if os.path.exists(file_path):
                                            file_size = os.path.getsize(file_path)
                                            st.text(f"📄 {file_name} ({file_size:,} bytes)")
                                        else:
                                            st.text(f"📄 {file_name} (file not found)")
                                    
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
                                            if os.path.exists(file_path):
                                                with open(file_path, 'rb') as f:
                                                    file_bytes = f.read()
                                                st.download_button(
                                                    label="💾 Download",
                                                    data=file_bytes,
                                                    file_name=file_name,
                                                    key=f"download_{file_name}"
                                                )
                                            else:
                                                st.error("File not found")
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

    # Loes Score Calculator Section
    display_loes_score_interface()
if __name__ == "__main__":
    main()
