# EOG-Based Eye Movement Classification System: Q&A

## Signal Processing

**Q1: What is the purpose of the bandpass filtering in this project?**  
A1: The bandpass filter (1-22Hz) removes noise and artifacts from the EOG signals. It eliminates low-frequency drift (baseline wander) and high-frequency noise while preserving the relevant eye movement information.

**Q2: Why is DC component removal performed after filtering?**  
A2: DC component removal (subtracting the mean) centers the signal around zero, eliminating offset bias that could affect feature extraction and classification. This makes the signal features more consistent across different recording sessions.

**Q3: What is the benefit of normalization in the signal processing pipeline?**  
A3: Normalization (to 0-1 range) standardizes signal amplitude across different recordings, making the system more robust to variations in signal strength due to electrode placement or individual differences.

## Feature Extraction

**Q4: What feature extraction methods are used in this project?**  
A4: Three feature extraction methods are used:
1. Autoregressive (AR) coefficients: Model the temporal dynamics of the signal
2. Wavelet coefficients: Capture frequency information at different scales
3. Statistical features: Extract simple statistical properties (mean, standard deviation, variance, energy)

**Q5: Why combine multiple feature types rather than using a single feature type?**  
A5: Different feature types capture complementary aspects of the EOG signals. AR coefficients model temporal patterns, wavelets capture multi-scale frequency information, and statistical features provide simple metrics of signal properties. Combined, they provide a more comprehensive representation of eye movements.

**Q6: What is the significance of combining horizontal and vertical EOG signals?**  
A6: Eye movements occur in both horizontal and vertical planes. By combining features from both signal channels, the system can better distinguish between different types of eye movements (e.g., left/right movements, blinks, etc.) that have characteristic patterns in both dimensions.

## Machine Learning

**Q7: Why is SVM chosen as the classification algorithm?**  
A7: SVM is well-suited for this application because:
- It performs well on medium-sized datasets
- It's effective in high-dimensional feature spaces
- It can handle complex, non-linear decision boundaries using kernels
- It's relatively robust against overfitting

**Q8: What does the RBF kernel do in the SVM implementation?**  
A8: The Radial Basis Function (RBF) kernel transforms the feature space to allow for non-linear decision boundaries between classes. This is important because the relationship between EOG features and eye movement classes is likely non-linear.

**Q9: How is class imbalance addressed in the classification?**  
A9: The SVM is configured with `class_weight='balanced'`, which automatically adjusts weights inversely proportional to class frequencies. This ensures that minority classes are not overwhelmed by majority classes during training.

## System Implementation

**Q10: How does the system process a new EOG signal pair for classification?**  
A10: The process follows these steps:
1. Load horizontal and vertical EOG signals
2. Preprocess signals (bandpass filter, DC removal, normalization)
3. Extract features (AR coefficients, wavelet coefficients, statistical features)
4. Combine features from both channels
5. Use the trained SVM model to predict the eye movement class
6. Take appropriate action based on the classification result

**Q11: How does the UI respond to different classification outcomes?**  
A11: The UI has specific actions tied to certain classifications:
- If "Right" is detected, it shows the next image
- If "Left" is detected, it shows the previous image
- For other classes (like "Blink"), it displays the class but takes no action

**Q12: What files are necessary for the system to function?**  
A12: The system requires:
- main.py: Core signal processing and classification algorithms
- ui.py: User interface implementation
- svm_model.pkl: The trained SVM classifier
- label_encoder.pkl: Encoder to map between numeric and text class labels
- Image files in the images directory

## Applications and Extensions

**Q13: What are potential applications for this EOG-based control system?**  
A13: Potential applications include:
- Assistive technology for people with motor disabilities
- Hands-free computer control
- Eye-controlled user interfaces
- Medical monitoring of eye movements
- Human-computer interaction research

**Q14: How could this system be extended to work in real-time?**  
A14: Real-time implementation would require:
- Continuous EOG signal acquisition hardware
- Buffering and windowing of incoming signals
- Optimized processing pipeline for low latency
- Parallel processing of signal acquisition and classification
- Interface adjustments for continuous control rather than discrete actions

**Q15: What challenges might arise in real-world deployment?**  
A15: Challenges include:
- Signal variability across users and sessions
- Electrode placement consistency
- User training and adaptation
- Distinguishing intentional from unintentional eye movements
- Fatigue effects during extended use
- Environmental and physiological noise interference 