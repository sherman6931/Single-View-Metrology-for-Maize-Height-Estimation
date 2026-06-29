# Maize Height Estimation Using Single-View Metrology

For the latest updates, raw source code, and interactive usage instructions, please visit our GitHub repository: https://github.com/sherman6931/Single-View-Metrology-for-Maize-Height-Estimation

This repository provides a reproducible demonstration package for maize plant-height estimation using single-view metrology (SVM). The released package includes an evaluation dataset, a trained maize top/base keypoint model, the geometric height-estimation code, a GUI demo, result tables, and a browser-based simulator for geometric checking.

This repository is released as the open-source project associated with the manuscript **End-to-End Intelligent Maize Plant Height Estimation: A Geometry-Constrained Single View Metrology Framework**. It is intended to provide the supporting code, released data, model weights, result files, and simulator needed for independent checking of the main computational workflow described in the paper.

Citation information will be updated after the paper is formally published.


## Repository structure

```text
maize-height-svm
│
├─ dataset
│  ├─ images
│  └─ metadata.csv
│
├─ results
│  ├─ height_results.csv
│  ├─ summary.csv
│  ├─ summary_by_height_group.csv
│  ├─ diagnostics.csv
│  └─ visualizations
│
├─ demo_app.py
├─ height_estimation.py
├─ maize_pose.pt
├─ maize_height_svm_simulator.html
├─ README.md
└─ requirements.txt
```

## Contents

- `dataset/images/`  
  Released maize images used by the demonstration package.

- `dataset/metadata.csv`  
  Metadata for each released image, including image name, camera height, measured plant height, and source image identifier.

- `maize_pose.pt`  
  Trained YOLO-Pose weight file for detecting maize top and stem-base keypoints.

- `height_estimation.py`  
  Geometric height-estimation script. It estimates the required geometric priors from image structures and calculates plant height using the SVM formulation.

- `demo_app.py`  
  GUI demonstration program. It automatically searches for the `.pt` model, reads `dataset/metadata.csv`, calls `height_estimation.py`, and visualizes the detected top/base keypoints and representative geometric cues.

- `results/height_results.csv`  
  Per-image height-estimation results, including estimated height, absolute error, relative error, selected horizon source, and visualization path.

- `results/summary.csv`  
  Overall evaluation metrics for the released dataset.

- `results/summary_by_height_group.csv`  
  Group-wise metrics according to measured plant-height groups.

- `results/diagnostics.csv`  
  Intermediate diagnostic information, including keypoints, bounding boxes, horizon/vanishing-point related quantities, and line-detection statistics.

- `results/visualizations/`  
  Visualization results corresponding to the released images.

- `maize_height_svm_simulator.html`  
  Browser-based SVM simulator with method comparison and full-factorial batch simulation functions.

## Environment setup

Python 3.9 or later is recommended.

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If GPU inference is not available, the demo program will attempt to fall back to CPU inference. CPU inference may be slower.

## Running the GUI demo

From the repository root, run:

```bash
python demo_app.py
```

Then click `Open Image` and select either:

```text
dataset/images
```

or the repository root folder. The program will automatically locate the released images and read the corresponding metadata.

Basic controls:

```text
Enter            Run measurement on the current image
Right / Space / N    Next image
Left / P             Previous image
```

In the GUI, the two calibration buttons keep the original interface style:

```text
Select Horizon (2 Parallel Lines)
Select Vertical (2 Parallel Lines)
```

For display clarity, the GUI shows two representative automatically detected structural cues. The internal geometric estimation still uses the full Hough/RANSAC/voting procedure implemented in `height_estimation.py`.

## Released evaluation results

The released dataset contains 42 images covering seven measured plant-height groups. The overall metrics for this released dataset are provided in:

```text
results/summary.csv
```

The corresponding per-image results are provided in:

```text
results/height_results.csv
```

The detailed metric values are available in the released result files. To keep this README focused on usage and reproducibility, specific numerical evaluation values are not repeated here.

These result files describe the released evaluation dataset in this repository.

## Running the simulator

The simulator is provided as:

```text
maize_height_svm_simulator.html
```

It can be opened directly in a modern web browser. An internet connection is required because the simulator loads Three.js and lil-gui from public CDNs.

The simulator provides a controlled virtual environment for checking the geometric SVM height-estimation component. It is not a replacement for the empirical greenhouse-image pipeline. In real images, maize top/base keypoints and scene-level geometric cues must still be extracted from image data; in this simulator, the corresponding image-plane coordinates are generated from the virtual 3D scene for controlled error analysis.

### Simulator method names

The simulator reports two methods:

- `Proposed SVM method`  
  The geometric height-estimation method proposed in this repository. It uses the camera height and four image-plane ordinates:

  ```text
  H_cam_m, y_top_px, y_bot_px, y_hor_px, y_v_px
  ```

  The implemented equation is:

  ```text
  H_obj = H_cam * (1 - ((y_top - y_hor) / (y_bot - y_hor)) * ((y_bot - y_v) / (y_top - y_v)))
  ```

  The virtual pole height is used only as ground truth for simulation evaluation; it is not used in the height calculation.

- `Calibration-dependent SVM`  
  A comparison method that reconstructs the bottom ground point and the top height from pixel rays using explicit focal length/FOV and pitch angle. The user-configurable focal, pitch, and camera-height bias settings are used to simulate calibration or pose-estimation errors.

### Basic interactive operation

After opening the HTML file:

```text
Mouse left-click       Place a pole marker on the ground plane
Mouse right-click      Remove a pole marker
Mouse middle-drag      Adjust camera orientation
Mouse wheel            Adjust FOV
W / A / S / D          Adjust the camera horizontal position
Space / Shift          Adjust the camera vertical position
```

At least one pole marker is required before SVM inputs and error metrics are displayed.

### Right-side control panel

The simulator control panel contains the following functional groups:

- `Camera Settings`  
  Adjusts the virtual camera pose and projection parameters. `Position Z (m)` in the simulator panel denotes the camera height above the ground plane, corresponding to `H_cam` in the SVM formulation. `Pitch (°)`, `Yaw (°)`, and `FOV (°)` control the viewing geometry.

- `Marker`  
  Adjusts the virtual pole marker, including `Pole H (m)` and ruler graduation size.

- `SVM Overlay`  
  Controls visual overlays, including the horizon/vanishing line, vertical vanishing point, top/base points, error text, and rounded-pixel calculation. `Use Rounded Pixel Inputs` rounds image coordinates to integer pixels to better mimic image-coordinate measurement.

- `SVM Method Comparison`  
  Controls whether the `Calibration-dependent SVM` comparison is shown and sets its assumed focal-length error, pitch bias, and camera-height error.

- `Simulation Evaluation`  
  Generates a random validation batch of pole markers using the current camera setting. `Random Batch Size` controls the number of generated poles, and `Min/Max GT Height (m)` controls the random ground-truth pole-height range.

- `Full-Factorial Batch`  
  Runs a Cartesian-product sweep over five variables:

  ```text
  Position Z × Pitch × Yaw × FOV × Pole H
  ```

  `Auto Dist Min/Max/Step` is not a swept experimental variable. It is only a nuisance placement range used to automatically search for a pole distance where the top and bottom endpoints are visible when possible.

- `Export Current View`  
  Exports the current annotated view as PNG, or exports current SVM samples as CSV/JSON.

### Full-factorial batch simulation

The full-factorial function is intended for reporting controlled simulation metrics. The default sweep ranges are:

```text
Position Z: 1.0 to 5.0 m, step 1.0 m
Pitch:      0 to 45 degrees, step 5 degrees
Yaw:        0 to 30 degrees, step 5 degrees
FOV:        15 to 60 degrees, step 5 degrees
Pole H:     0.6 to 2.7 m, step 0.1 m
```

These defaults generate 77,000 Cartesian-product rows before visibility filtering.

Recommended operation:

```text
1. Open the Full-Factorial Batch panel.
2. Check or adjust the min/max/step values for Position Z, Pitch, Yaw, FOV, and Pole H.
3. Click Preview Combination Count to confirm the number of generated combinations.
4. Click Generate Full Grid Metrics to compute and display the on-screen summary.
5. Click Export Full Grid CSV to export detailed per-row results and a summary JSON file.
6. Use Export Full Grid JSON only when a full detailed JSON export is needed; it may be large for dense sweeps.
```

The simulator keeps every Cartesian-product row. If no placement can keep the pole top and bottom inside the image frame for a given camera setting, that row is still retained and marked with visibility flags.

For reporting, the simulator panel recommends `Visible-only` metrics because they exclude rows where the pole top or bottom is outside the image frame. The relevant exported visibility field is:

```text
both_inside_frame
```

### Output metrics

The simulator reports the following metrics for both methods:

```text
MRE       Mean Relative Error
RMSE      Root Mean Square Error
Max RE    Maximum Relative Error
≤1% ratio Proportion of valid visible samples with relative error not higher than 1%
```

For full-factorial simulation, the output panel also reports:

```text
Total rows       Number of Cartesian-product combinations generated
Visible rows     Number of rows where both top and bottom endpoints are inside the image frame
Visible ratio    Visible rows / total rows
Winner by visible MRE  Method with the lower visible-only MRE
```

### Exported files

Current-view export functions:

```text
Export Annotated PNG   Save the current rendered view with SVM overlay
Export SVM JSON        Export current pole samples, SVM inputs, method settings, and summary metrics
Export SVM CSV         Export current pole samples and per-sample comparison results
```

Full-factorial export functions:

```text
Export Full Grid CSV   Export detailed full-factorial rows and a separate summary JSON
Export Full Grid JSON  Export summary and detailed sample objects in one JSON file
```

Important exported fields include:

```text
H_cam_m
y_top_px, y_bot_px, y_hor_px, y_v_px
proposed_estimated_height_m
proposed_relative_error_percent
calibration_dependent_svm_estimated_height_m
calibration_dependent_svm_relative_error_percent
ground_truth_height_m
top_inside_frame, bottom_inside_frame, both_inside_frame
```

### Scope of the simulator results

The simulator is designed to check the geometric formulation under idealized, controllable virtual scenes. It is useful for verifying parameter sensitivity, visibility constraints, and the difference between the proposed SVM method and the calibration-dependent SVM method. The real-image evaluation results in `results/` should be used for the released maize-image dataset, while the simulator results should be interpreted as controlled geometric validation.

## Notes

- The camera height and measured plant height for each released image are stored in `dataset/metadata.csv`.
- The GUI demo automatically searches for `maize_pose.pt` and `height_estimation.py` in the repository.
- The released package is intended to support independent checking of the main workflow and result files.
- The simulator is included to illustrate and validate the geometric formulation under controlled virtual conditions.
