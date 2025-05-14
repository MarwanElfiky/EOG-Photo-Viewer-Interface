import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import joblib
import numpy as np
from collections import deque
from final_classifier import (
    bandpass_filter, normalize_signal, adaptive_trim_signal, 
    standardize_signal_length, extract_super_features
)

# Global variables
root = None
images = []
current_index = 0
is_detecting = False
last_detection_time = 0
detection_cooldown = 1.5  # seconds
invert_directions = True  # Invert left/right directions

# Consensus detection variables
detection_history = deque(maxlen=5)  # Store last 5 predictions
min_consensus = 3  # Need at least 3 matching predictions
confidence_threshold = 0.6  # Minimum confidence

# UI elements
image_label = None
status_label = None
prev_button = None
next_button = None
start_button = None
debug_window = None
invert_button = None

# Classifier 
model = None
label_encoder = None

def load_model():
    """Load the EOG classifier model"""
    global model, label_encoder
    
    try:
        model = joblib.load('eog_classifier_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        print("Classifier loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading classifier: {e}")
        return False

def load_images(folder_path=None):
    """Load images from a folder"""
    global images, current_index
    
    if not folder_path:
        folder_path = filedialog.askdirectory(title="Select folder with images")
        if not folder_path:
            return False
    
    # Supported image formats
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    try:
        # Get all image files
        image_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if any(f.lower().endswith(ext) for ext in extensions)
        ]
        
        if not image_files:
            messagebox.showinfo("No Images", "No images found in the selected folder.")
            return False
        
        # Sort images by name
        images = sorted(image_files)
        current_index = 0
        return True
    
    except Exception as e:
        messagebox.showerror("Error", f"Error loading images: {e}")
        return False

def toggle_direction_inversion():
    """Toggle the inversion of left/right direction detection"""
    global invert_directions, invert_button
    
    invert_directions = not invert_directions
    
    if invert_directions:
        invert_button.config(text="Directions: INVERTED", bg="#E91E63")
        status_label.config(text="LEFT eye movement = RIGHT action, RIGHT eye movement = LEFT action")
    else:
        invert_button.config(text="Directions: NORMAL", bg="#4CAF50")
        status_label.config(text="LEFT eye movement = LEFT action, RIGHT eye movement = RIGHT action")
    
    # Clear existing detection history with the new setting
    detection_history.clear()
    update_debug_info(f"Direction inversion: {invert_directions}")

def display_current_image():
    """Show the current image in the viewer"""
    global image_label, status_label, current_index, images
    
    if not images:
        status_label.config(text="No images loaded")
        return
    
    try:
        # Load and resize the image
        img = Image.open(images[current_index])
        
        # Calculate size to maintain aspect ratio
        window_width = root.winfo_width() - 40
        window_height = root.winfo_height() - 120
        img_width, img_height = img.size
        
        # Determine scaling factor
        width_ratio = window_width / img_width
        height_ratio = window_height / img_height
        ratio = min(width_ratio, height_ratio)
        
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        # Update the image and status
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference
        
        # Update status with image info
        filename = os.path.basename(images[current_index])
        status_label.config(text=f"Image {current_index + 1} of {len(images)}: {filename}")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error displaying image: {e}")

def previous_image():
    """Show the previous image"""
    global current_index, images, prev_button
    
    if not images:
        return
    
    # Highlight the button
    highlight_button(prev_button)
    
    # Update the index with wraparound
    current_index = (current_index - 1) % len(images)
    display_current_image()

def next_image():
    """Show the next image"""
    global current_index, images, next_button
    
    if not images:
        return
    
    # Highlight the button
    highlight_button(next_button)
    
    # Update the index with wraparound
    current_index = (current_index + 1) % len(images)
    display_current_image()

def highlight_button(button, duration=300):
    """Highlight a button temporarily"""
    original_color = button.cget("background")
    button.config(background="#FFC107")  # Yellow highlight
    root.after(duration, lambda: button.config(background=original_color))

def process_eog_signal(signal):
    """Process an EOG signal and predict the movement class with confidence check"""
    if model is None or label_encoder is None:
        print("Model not loaded")
        return None, 0
    
    try:
        # Preprocess the signal
        signal = np.array(signal)
        signal = np.nan_to_num(signal)
        filtered = bandpass_filter(signal, 0.1, 40, 250, order=6)
        trimmed = adaptive_trim_signal(filtered)
        resampled = standardize_signal_length(trimmed, target_length=1000)
        normalized = normalize_signal(resampled)
        
        # Extract features and predict
        features = extract_super_features(normalized)
        features = features.reshape(1, -1)
        
        # Get prediction and probability
        predicted_idx = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = probabilities[predicted_idx]
        
        predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
        
        # Only consider left/right for photo viewing
        if predicted_class not in ['left', 'right']:
            return None, confidence
        
        # Apply direction inversion if enabled
        if invert_directions and predicted_class in ['left', 'right']:
            predicted_class = 'left' if predicted_class == 'right' else 'right'
            
        # Only return the prediction if confidence is above threshold
        if confidence >= confidence_threshold:
            return predicted_class, confidence
        else:
            return None, confidence
    
    except Exception as e:
        print(f"Error processing signal: {e}")
        return None, 0

def check_consensus():
    """Check if we have consensus on a movement"""
    global detection_history
    
    if len(detection_history) < min_consensus:
        return None
    
    # Count occurrences of each direction
    left_count = sum(1 for d in detection_history if d == 'left')
    right_count = sum(1 for d in detection_history if d == 'right')
    
    # Need at least min_consensus votes for a direction
    if left_count >= min_consensus:
        detection_history.clear()  # Reset after consensus
        update_debug_info(f"CONSENSUS: LEFT ({left_count}/{len(detection_history)})")
        return 'left'
    elif right_count >= min_consensus:
        detection_history.clear()  # Reset after consensus
        update_debug_info(f"CONSENSUS: RIGHT ({right_count}/{len(detection_history)})")
        return 'right'
    
    return None

def handle_movement(movement, confidence=0):
    """Handle a detected eye movement with confidence check"""
    global last_detection_time, status_label, detection_history
    
    # Add to detection history
    if movement:
        detection_history.append(movement)
        update_debug_info(f"Added {movement} to history: {list(detection_history)}")
    
    # Check for consensus
    consensus_movement = check_consensus()
    if not consensus_movement:
        return
    
    # Apply cooldown to prevent rapid successive actions
    current_time = import_time().time()
    if current_time - last_detection_time < detection_cooldown:
        return
    
    last_detection_time = current_time
    
    # Update status with confirmed movement
    status_label.config(text=f"Confirmed: {consensus_movement.upper()} movement")
    
    # Handle the movement
    if consensus_movement == 'left':
        previous_image()
    elif consensus_movement == 'right':
        next_image()

def import_time():
    """Lazy import time module to avoid issues"""
    import time
    return time

def simulate_detection():
    """Simulate EOG detections for testing with occasional false negatives"""
    global is_detecting, status_label
    
    if not is_detecting:
        return
    
    # Multiple detection classes including noise
    movements = ['left', 'right', None, 'noise']
    weights = [0.35, 0.35, 0.2, 0.1]  # 20% no detection, 10% noise
    
    detected = np.random.choice(movements, p=weights)
    
    if detected == 'noise':
        # Simulate alternating noise
        noise_class = np.random.choice(['up', 'down', 'blink'])
        confidence = np.random.uniform(0.4, 0.6)
        update_debug_info(f"Noise detected: {noise_class} (conf: {confidence:.2f})")
        status_label.config(text=f"Filtered: {noise_class} (not left/right)")
    elif detected:
        # Simulate varying confidence levels
        confidence = np.random.uniform(0.6, 0.95)
        update_debug_info(f"Detected: {detected} (conf: {confidence:.2f})")
        status_label.config(text=f"Detected: {detected} (Conf: {confidence:.2f})")
        handle_movement(detected, confidence)
    else:
        # Show that a signal was processed but rejected due to low confidence
        low_conf = np.random.uniform(0.3, 0.55)
        update_debug_info(f"Low confidence: {low_conf:.2f}")
        status_label.config(text=f"Signal too weak (conf: {low_conf:.2f})")
    
    # Schedule next detection if still running
    if is_detecting:
        # Random interval between 1 and 2 seconds
        interval = np.random.uniform(1000, 2000)
        root.after(int(interval), simulate_detection)

def toggle_detection():
    """Toggle EOG detection on/off"""
    global is_detecting, start_button, detection_history
    
    if not model:
        messagebox.showerror("Error", "Classifier not loaded. Cannot start detection.")
        return
    
    is_detecting = not is_detecting
    detection_history.clear()
    
    if is_detecting:
        start_button.config(text="Stop Detection", bg="#E91E63")
        
        # Reminder of current direction setting
        if invert_directions:
            status_label.config(text="EOG detection active - LEFT eye = RIGHT action, RIGHT eye = LEFT action")
        else:
            status_label.config(text="EOG detection active - directions match eye movements")
            
        simulate_detection()  # Start simulating detections
    else:
        start_button.config(text="Start Detection", bg="#4CAF50")
        status_label.config(text="EOG detection stopped")

def adjust_sensitivity():
    """Allow user to adjust detection sensitivity"""
    global confidence_threshold, min_consensus, detection_cooldown
    
    # Create settings window
    settings = tk.Toplevel(root)
    settings.title("Detection Settings")
    settings.geometry("400x300")
    settings.grab_set()
    
    tk.Label(settings, text="Adjust EOG Detection Settings", font=("Arial", 12, "bold")).pack(pady=10)
    
    # Confidence threshold
    conf_frame = tk.Frame(settings)
    conf_frame.pack(fill=tk.X, padx=20, pady=10)
    tk.Label(conf_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
    conf_slider = tk.Scale(conf_frame, from_=0.4, to=0.9, resolution=0.05, 
                         orient=tk.HORIZONTAL, length=200)
    conf_slider.set(confidence_threshold)
    conf_slider.pack(side=tk.RIGHT)
    
    # Consensus required
    cons_frame = tk.Frame(settings)
    cons_frame.pack(fill=tk.X, padx=20, pady=10)
    tk.Label(cons_frame, text="Required Consensus:").pack(side=tk.LEFT)
    cons_slider = tk.Scale(cons_frame, from_=2, to=5, resolution=1, 
                          orient=tk.HORIZONTAL, length=200)
    cons_slider.set(min_consensus)
    cons_slider.pack(side=tk.RIGHT)
    
    # Cooldown period
    cool_frame = tk.Frame(settings)
    cool_frame.pack(fill=tk.X, padx=20, pady=10)
    tk.Label(cool_frame, text="Cooldown Period (s):").pack(side=tk.LEFT)
    cool_slider = tk.Scale(cool_frame, from_=0.5, to=3.0, resolution=0.1, 
                          orient=tk.HORIZONTAL, length=200)
    cool_slider.set(detection_cooldown)
    cool_slider.pack(side=tk.RIGHT)
    
    # Description
    desc = (
        "Higher confidence = fewer false positives but less sensitivity\n"
        "Higher consensus = more reliable but requires more consistent signals\n"
        "Higher cooldown = fewer rapid transitions but slower response"
    )
    tk.Label(settings, text=desc, justify=tk.LEFT, font=("Arial", 9)).pack(padx=20, pady=10)
    
    # Apply button
    def apply_settings():
        global confidence_threshold, min_consensus, detection_cooldown
        confidence_threshold = conf_slider.get()
        min_consensus = int(cons_slider.get())
        detection_cooldown = cool_slider.get()
        detection_history.clear()  # Reset history with new settings
        status_label.config(text=f"Settings updated: Conf={confidence_threshold}, Cons={min_consensus}, Cool={detection_cooldown}s")
        settings.destroy()
    
    tk.Button(settings, text="Apply", command=apply_settings, 
             bg="#4CAF50", fg="white", font=("Arial", 11)).pack(pady=10)

def toggle_debug_window():
    """Toggle debug window visibility"""
    global debug_window
    
    if debug_window and debug_window.winfo_exists():
        debug_window.destroy()
        debug_window = None
    else:
        create_debug_window()

def create_debug_window():
    """Create a debug window to show detection details"""
    global debug_window
    
    debug_window = tk.Toplevel(root)
    debug_window.title("EOG Detection Debug")
    debug_window.geometry("500x300")
    
    # Create text widget for debug info
    debug_text = tk.Text(debug_window, wrap=tk.WORD, height=15, width=60)
    debug_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    # Add scrollbar
    scrollbar = tk.Scrollbar(debug_text)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    debug_text.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=debug_text.yview)
    
    # Store reference to the text widget
    debug_window.text = debug_text
    
    # Initial text
    debug_text.insert(tk.END, "EOG Detection Debug Information\n")
    debug_text.insert(tk.END, "--------------------------------\n")
    debug_text.insert(tk.END, f"Confidence threshold: {confidence_threshold}\n")
    debug_text.insert(tk.END, f"Required consensus: {min_consensus}\n")
    debug_text.insert(tk.END, f"Cooldown period: {detection_cooldown}s\n")
    debug_text.insert(tk.END, f"Direction inversion: {invert_directions}\n\n")
    debug_text.see(tk.END)

def update_debug_info(message):
    """Update the debug window with new information"""
    if debug_window and debug_window.winfo_exists():
        import_time = __import__('time')
        timestamp = import_time.strftime("%H:%M:%S", import_time.localtime())
        debug_window.text.insert(tk.END, f"[{timestamp}] {message}\n")
        debug_window.text.see(tk.END)

def create_ui():
    """Create the user interface"""
    global root, image_label, status_label, prev_button, next_button, start_button, invert_button
    
    # Create main window
    root = tk.Tk()
    root.title("EOG Photo Viewer")
    root.geometry("900x700")
    root.minsize(800, 600)
    
    # Create main frame with padding
    main_frame = tk.Frame(root, bg="#F0F0F0", padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Top control panel
    control_panel = tk.Frame(main_frame, bg="#E0E0E0", height=40)
    control_panel.pack(fill=tk.X, pady=(0, 10))
    
    # Load images button
    load_button = tk.Button(control_panel, text="Load Images", 
                            command=lambda: load_images() and display_current_image(),
                            bg="#2196F3", fg="white", font=("Arial", 11))
    load_button.pack(side=tk.LEFT, padx=10, pady=5)
    
    # Start/stop detection button
    start_button = tk.Button(control_panel, text="Start Detection", 
                             command=toggle_detection,
                             bg="#4CAF50", fg="white", font=("Arial", 11))
    start_button.pack(side=tk.LEFT, padx=10, pady=5)
    
    # Add sensitivity adjustment button
    sensitivity_button = tk.Button(control_panel, text="Adjust Settings", 
                                  command=adjust_sensitivity,
                                  bg="#9C27B0", fg="white", font=("Arial", 11))
    sensitivity_button.pack(side=tk.LEFT, padx=10, pady=5)
    
    # Add debug window toggle button
    debug_button = tk.Button(control_panel, text="Debug Window", 
                            command=toggle_debug_window,
                            bg="#FF9800", fg="white", font=("Arial", 11))
    debug_button.pack(side=tk.LEFT, padx=10, pady=5)
    
    # Add direction inversion button (since the classifier seems to be reversed)
    invert_button = tk.Button(control_panel, text="Directions: INVERTED", 
                             command=toggle_direction_inversion,
                             bg="#E91E63", fg="white", font=("Arial", 11))
    invert_button.pack(side=tk.LEFT, padx=10, pady=5)
    
    # Status label
    status_label = tk.Label(control_panel, text="Ready to load images", 
                            bg="#E0E0E0", font=("Arial", 11))
    status_label.pack(side=tk.RIGHT, padx=10, pady=5)
    
    # Image display area
    image_area = tk.Frame(main_frame, bg="white")
    image_area.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Label to display the image
    image_label = tk.Label(image_area, bg="white")
    image_label.pack(fill=tk.BOTH, expand=True)
    
    # Navigation panel
    nav_panel = tk.Frame(main_frame, bg="#E0E0E0", height=60)
    nav_panel.pack(fill=tk.X, pady=(10, 0))
    
    # Previous button
    prev_button = tk.Button(nav_panel, text="← Previous", command=previous_image,
                           bg="#FF5722", fg="white", font=("Arial", 12, "bold"),
                           width=15, height=2)
    prev_button.pack(side=tk.LEFT, padx=40, pady=10)
    
    # Next button
    next_button = tk.Button(nav_panel, text="Next →", command=next_image,
                           bg="#2196F3", fg="white", font=("Arial", 12, "bold"),
                           width=15, height=2)
    next_button.pack(side=tk.RIGHT, padx=40, pady=10)
    
    # Keyboard bindings
    root.bind('<Left>', lambda e: previous_image())
    root.bind('<Right>', lambda e: next_image())
    
    # Window resize handler
    def on_resize(event):
        # Wait a bit to let the window finish resizing
        root.after(100, display_current_image)
    
    root.bind("<Configure>", on_resize)
    
    return root

def main():
    """Main entry point for the application"""
    load_model()
    app = create_ui()
    app.mainloop()

if __name__ == "__main__":
    main() 