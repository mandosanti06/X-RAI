# X-RAI: Bone Fracture Detection from X-ray Images

## Overview  
X-RAI is a deep learning project that automatically detects bone fractures from musculoskeletal X-ray images using the MURA v1.1 dataset. The project employs a lightweight convolutional neural network (MobileNetV2) to classify each image as "Fractured" or "Not Fractured."

Google colabnis located at:[ **Google Colab**](https://colab.research.google.com/drive/1YwzeA-ePgQhsQlSw-GwWB0M2pPhOz_2n?usp=sharing)

---

## Features  
- **Fracture Classification**: Achieves up to ~80% accuracy on the validation subset.  
- **Fast Training**: Utilizes MobileNetV2 with resized images for efficient training.  
- **Model Interpretability**: Includes Grad-CAM visualizations to highlight regions influencing the model's decisions.  
- **Label Overlays**: Displays predictions with probabilities for better interpretation.  

---

## Stack  
- **Dataset**: MURA v1.1 (Stanford ML Group)  
- **Model**: MobileNetV2 (pretrained on ImageNet)  
- **Frameworks**: PyTorch, Torchvision  
- **Visualization**: Grad-CAM, Matplotlib  
- **Environment**: Google Colab (T4 GPU) / Kaggle (GPU)  

---

## Setup for Google Colab  
To ensure optimal performance, use a **T4 GPU** on Google Colab:  
1. Open the notebook in Google Colab.  
2. Navigate to **Runtime → Change runtime type**.  
3. Select **T4 GPU** under *Hardware Accelerator*.  
4. Click **Save**.  

Verify GPU availability by running:  
```python
import torch
print("GPU Enabled:", torch.cuda.is_available())
```

---

## Installation  
Install dependencies in Colab:  
```bash
!pip install kagglehub torch torchvision matplotlib scikit-learn tqdm grad-cam
```

---

## Usage  
1. **Dataset Download**: The notebook automatically downloads the MURA v1.1 dataset using KaggleHub.  
2. **Body Part Selection**: Modify the `bodypart` variable (e.g., `"humerus"`, `"wrist"`) to analyze specific body parts.  
3. **Training**: Execute the training loop (5 epochs by default). Adjust `num_epochs` for extended training.  
4. **Evaluation**: Validation metrics (precision, recall, F1-score) are printed post-training.  
5. **Visualization**: Grad-CAM highlights model focus areas on a random validation sample.  

---

## Example Output  
The notebook displays:  
- A classification report with precision, recall, and F1-score.  
- Side-by-side images of the original X-ray and Grad-CAM heatmap, annotated with actual and predicted labels.  

---

## Notes  
- The T4 GPU accelerates training significantly compared to CPU-only mode.  
- Free Colab tiers may occasionally assign other GPUs, but T4 is prioritized in most sessions.  

---

## License  
This project is open-source and available for educational and research purposes. Cite the MURA dataset and relevant libraries if used in your work.  

---

## Acknowledgments  
- **MURA Dataset**: Stanford ML Group  
- **Libraries**: PyTorch, Torchvision, Grad-CAM, Matplotlib  

For questions or contributions, open an issue or contact the author.
