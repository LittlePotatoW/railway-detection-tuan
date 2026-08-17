[English](README.md) | [中文](README.zn.md)

# Railway Detection "Tuan" of USTB

Unsupervised railway track defect detection based on graphics and statistics

## Abstract

Railway track defects come in many forms, while the number of images available for training is scarce, and they can pose serious safety hazards. To address this problem, we propose an unsupervised detection method based on graphics and statistics that builds a highly sensitive detection model from only a small number of normal track images, without requiring any defective samples for annotation.

Under a fixed light source and a fixed camera viewpoint, the surface of a metal rail exhibits linear highlights nearly parallel to the track, while defects disrupt the brightness and shape of these highlights. Based on this observation, our method first performs geometric correction to automatically align the track to the center of the image. It then applies an FFT-based directional filter in the frequency domain to suppress the linear highlight texture while preserving defect structures. Next, a "normal track model" is built by computing the per-pixel mean μ and standard deviation σ from normal track samples. During detection, a per-pixel Z-score is computed for the input image; after thresholding, morphological cleanup, and connected-component analysis, the defect locations are output as bounding boxes.

Our model is compact, computationally simple, and highly sensitive, making it suitable for deployment on edge devices such as inspection carts. It can also serve as a fast on-device pre-filter, working together with an image recognition model to classify defects while balancing recall and accuracy.

![Defect detection example](https://github.com/user-attachments/assets/ab771fbf-8e84-4b5d-a909-f759b562c86c)

## Notes

- The algorithm is based entirely on the assumptions that the camera position, track type, and lighting conditions remain unchanged, and the model is highly sensitive. Therefore, all images used for training must be captured on-site with the actual inspection setup; any change to the camera, light source, or cart height requires recollecting photos and retraining the model.

## License

This project is distributed under a custom license based on the MIT License. It requires attribution when the code is cited or used, and prohibits using it, in whole or in part, in competitions held before December 1, 2026.
See [LICENSE](LICENSE) for details.
