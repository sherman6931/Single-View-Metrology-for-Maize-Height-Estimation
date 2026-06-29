import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import math
import re
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd



try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None









YOLO_MODEL_PATH = ""
GEOM_METHOD_SCRIPT_PATH = ""



IMGSZ = 1600
YOLO_CONF = 0.20
YOLO_IOU = 0.70
MAX_DET_CANDIDATES = 20
USE_CENTER_PRIORITY = False
YOLO_DEVICE = 0



KEYPOINT_INDEX_TOP = 0
KEYPOINT_INDEX_BASE = 1


FOLDER_IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


ERASE_PLANT_BBOX_BEFORE_HOUGH = True
ERASE_BBOX_EXPAND_RATIO = 0.12

USE_CLAHE = True
GAUSSIAN_BLUR_KSIZE = 5
CANNY_SIGMA = 0.33

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180.0
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 90
HOUGH_MAX_LINE_GAP = 40
MIN_LINE_LENGTH_PX = 80

RANSAC_ITER = 1600
RANSAC_DIST_THRESH_PX = 14.0
MAX_VP_ABS_FACTOR = 60

Y_HOR_MIN_FACTOR = -0.65
Y_HOR_MAX_FACTOR = 1.35
Y_V_MAX_ABS_FACTOR = 25.0

DIAG_MIN_ANGLE = 8
DIAG_MAX_ANGLE = 82
VERTICAL_MIN_ANGLE = 70
VERTICAL_MAX_ANGLE = 110
HORIZONTAL_TOL_ANGLE = 14
MAX_HORIZON_TILT_DEG = 25

USE_STRUCTURAL_ROI = True
MASK_PLANT_BBOX_AFTER_HOUGH = True
BBOX_EXPAND_RATIO_AFTER_HOUGH = 0.12

MIN_DIAG_INLIERS = 2
MIN_VERTICAL_INLIERS = 4
MIN_VERTICAL_INLIER_RATIO = 0.020


MIN_VALID_HEIGHT_M = 0.05
MAX_VALID_HEIGHT_M = 5.00






def get_app_dir() -> Path:
    
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd().resolve()


def unique_existing_dirs(paths):
    seen = set()
    out = []
    for p in paths:
        try:
            p = Path(p).resolve()
        except Exception:
            continue
        if p.exists() and p.is_dir() and str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def candidate_release_dirs() -> list[Path]:
    




    app_dir = get_app_dir()
    return unique_existing_dirs([
        app_dir,
        app_dir.parent,
        Path.cwd(),
        Path.cwd().parent,
    ])


def resolve_yolo_model_path() -> Path:
    








    if YOLO_MODEL_PATH:
        p = Path(YOLO_MODEL_PATH)
        if p.exists() and p.is_file():
            return p
        raise RuntimeError(f"Specified YOLO_MODEL_PATH does not exist:\n{p}")

    exact_names = [
        "maize_pose.pt",
        "yolov11n_pose_maize_topbase.pt",
        "best.pt",
        "modelsmaize_pose.pt",  
    ]

    search_dirs = []
    for root in candidate_release_dirs():
        search_dirs.extend([
            root,
            root / "models",
            root / "weights",
        ])
    search_dirs = unique_existing_dirs(search_dirs)

    for d in search_dirs:
        for name in exact_names:
            p = d / name
            if p.exists() and p.is_file():
                return p

    all_pts = []
    for d in search_dirs:
        all_pts.extend(sorted(d.glob("*.pt")))

    if len(all_pts) == 1:
        return all_pts[0]
    if len(all_pts) > 1:
        preferred = [p for p in all_pts if "maize" in p.name.lower() or "pose" in p.name.lower()]
        if preferred:
            return preferred[0]
        return all_pts[0]

    raise RuntimeError(
        "Cannot find YOLO .pt model automatically.\n"
        "Please put the weight file in the same folder as demo_app.py, "
        "or in ./models/, for example: models/maize_pose.pt"
    )


def resolve_geom_method_script_path() -> Path:
    






    if GEOM_METHOD_SCRIPT_PATH:
        p = Path(GEOM_METHOD_SCRIPT_PATH)
        if p.exists() and p.is_file():
            return p
        raise RuntimeError(f"Specified GEOM_METHOD_SCRIPT_PATH does not exist:\n{p}")

    exact_names = [
        "geom_hough_voting_ransac_v1_height.py",
        "height_estimation.py",
        "svm_hough_ransac_height_estimation.py",
        "srcheight_estimation.py",  
    ]

    search_dirs = []
    for root in candidate_release_dirs():
        search_dirs.extend([
            root,
            root / "src",
        ])
    search_dirs = unique_existing_dirs(search_dirs)

    for d in search_dirs:
        for name in exact_names:
            p = d / name
            if p.exists() and p.is_file() and p.resolve() != Path(__file__).resolve():
                return p

    
    for d in search_dirs:
        candidates = []
        for p in d.glob("*.py"):
            name = p.name.lower()
            if p.resolve() == Path(__file__).resolve():
                continue
            if ("height" in name and ("hough" in name or "ransac" in name or "estimation" in name)):
                candidates.append(p)
        if candidates:
            return sorted(candidates, key=lambda x: x.name.lower())[0]

    raise RuntimeError(
        "Cannot find the geometric height-estimation script automatically.\n"
        "Please put geom_hough_voting_ransac_v1_height.py in the same folder as demo_app.py, "
        "or put height_estimation.py in ./src/."
    )


def resolve_image_folder_from_selected_folder(selected_folder: Path) -> Path:
    


    selected_folder = Path(selected_folder)
    candidates = [
        selected_folder,
        selected_folder / "dataset" / "images",
        selected_folder / "images",
    ]
    
    if selected_folder.name.lower() == "dataset":
        candidates.insert(0, selected_folder / "images")

    for c in candidates:
        if c.exists() and c.is_dir():
            has_img = any(p.is_file() and p.suffix.lower() in FOLDER_IMAGE_SUFFIXES for p in c.rglob("*"))
            if has_img:
                return c
    return selected_folder


def find_metadata_csv_for_image_folder(image_folder: Path) -> Path | None:
    




    image_folder = Path(image_folder).resolve()
    app_dir = get_app_dir()

    candidates = [
        image_folder.parent / "metadata.csv",                 
        image_folder.parent.parent / "dataset" / "metadata.csv",
        image_folder / "metadata.csv",
        app_dir / "dataset" / "metadata.csv",
        app_dir.parent / "dataset" / "metadata.csv",
        Path.cwd() / "dataset" / "metadata.csv",
    ]

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            pass
    return None


class RobustMetrologyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Single View Metrology Pro - Robust Edition")
        self.root.geometry("1280x850")

        
        self.img_path = None
        self.folder_path = None
        self.metadata_path = None
        self.metadata_lookup = {}
        self.image_list = []
        self.image_index = 0
        self.current_camera_height_cm = None
        self.current_true_height_cm = None
        self.current_height_parse_source = ""
        self.pil_image = None  
        self.tk_image = None  
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.img_w = 0
        self.img_h = 0

        
        self.principal_point = (0, 0)  
        self.focal_length = 0.0
        self.pitch = 0.0  
        self.horizon_y = 0.0  
        self.vz_y = 0.0  

        
        self.state = "IDLE"
        self.clicks = []
        self.lines_horizon = []  
        self.lines_vertical = []  

        
        self.yolo_model = None
        self.auto_detection = None
        self.auto_priors = None
        self.auto_candidate = None
        self.auto_height_result = None
        self._computed_auto_result = None

        self._setup_ui()

        
        self.canvas.bind("<Configure>", self.resize_image)

        
        
        
        
        
        self.root.bind_all("<Right>", lambda e: self.next_image())
        self.root.bind_all("<Down>", lambda e: self.next_image())
        self.root.bind_all("<space>", lambda e: self.next_image())
        self.root.bind_all("n", lambda e: self.next_image())
        self.root.bind_all("N", lambda e: self.next_image())
        self.root.bind_all("<Next>", lambda e: self.next_image())

        self.root.bind_all("<Left>", lambda e: self.prev_image())
        self.root.bind_all("<Up>", lambda e: self.prev_image())
        self.root.bind_all("<BackSpace>", lambda e: self.prev_image())
        self.root.bind_all("p", lambda e: self.prev_image())
        self.root.bind_all("P", lambda e: self.prev_image())
        self.root.bind_all("<Prior>", lambda e: self.prev_image())

        self.root.bind_all("<Return>", lambda e: self.quick_run_current_image())

    def _setup_ui(self):
        
        panel = tk.Frame(self.root, width=340, bg="#f0f0f0", padx=15, pady=15)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        panel.pack_propagate(False)

        
        style_h1 = ("Helvetica", 12, "bold")
        bg_color = "#f0f0f0"

        
        tk.Label(panel, text="Step 1: Load & Setup", font=style_h1, bg=bg_color).pack(anchor="w", pady=5)
        tk.Button(panel, text="Open Image", command=self.load_image, bg="white", height=2).pack(fill=tk.X)

        tk.Label(panel, text="Camera Height Hc (meters):", bg=bg_color).pack(anchor="w", pady=(10, 0))
        self.entry_hc = tk.Entry(panel)
        self.entry_hc.insert(0, "1.6")
        self.entry_hc.pack(fill=tk.X)

        ttk.Separator(panel, orient='horizontal').pack(fill='x', pady=15)

        
        tk.Label(panel, text="Step 2: Calibration", font=style_h1, bg=bg_color).pack(anchor="w", pady=5)

        self.mode_var = tk.StringVar(value="auto")
        tk.Radiobutton(panel, text="A. Vanishing Points (Auto)", variable=self.mode_var, value="auto",
                       bg=bg_color, command=self.toggle_mode).pack(anchor="w")

        
        self.btn_sel_hor = tk.Button(panel, text="Select Horizon (2 Parallel Lines)",
                                     command=self.select_horizon_button_action, state="disabled")
        self.btn_sel_hor.pack(fill=tk.X, pady=2)

        self.btn_sel_ver = tk.Button(panel, text="Select Vertical (2 Parallel Lines)",
                                     command=self.select_vertical_button_action, state="disabled")
        self.btn_sel_ver.pack(fill=tk.X, pady=2)

        
        self.btn_reset_calib = tk.Button(panel, text="Reset Selection", command=self.reset_calibration_selection,
                                         bg="#ffebee", state="disabled")
        self.btn_reset_calib.pack(fill=tk.X, pady=(5, 10))

        tk.Button(panel, text=">> Calculate Focal & Pitch <<", command=self.calculate_camera_params,
                  bg="#d1ecf1", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=5)

        
        tk.Radiobutton(panel, text="B. Manual Input", variable=self.mode_var, value="manual",
                       bg=bg_color, command=self.toggle_mode).pack(anchor="w", pady=(10, 0))

        frm_manual = tk.Frame(panel, bg=bg_color)
        frm_manual.pack(fill=tk.X, padx=5)
        tk.Label(frm_manual, text="Focal(px):", bg=bg_color).grid(row=0, column=0)
        self.ent_f = tk.Entry(frm_manual, width=10)
        self.ent_f.grid(row=0, column=1, padx=5)
        tk.Label(frm_manual, text="Pitch(deg):", bg=bg_color).grid(row=1, column=0)
        self.ent_pitch = tk.Entry(frm_manual, width=10)
        self.ent_pitch.grid(row=1, column=1, padx=5)

        ttk.Separator(panel, orient='horizontal').pack(fill='x', pady=15)

        
        tk.Label(panel, text="Step 3: Measure", font=style_h1, bg=bg_color).pack(anchor="w", pady=5)
        self.btn_measure = tk.Button(panel, text="Measure Object (Head -> Feet)",
                                     command=self.measure_button_action,
                                     bg="#d4edda", state="disabled", height=2)
        self.btn_measure.pack(fill=tk.X, pady=5)

        
        self.btn_clear_meas = tk.Button(panel, text="Clear Measurements", command=self.clear_measurements,
                                        bg="#fff3cd", state="disabled")
        self.btn_clear_meas.pack(fill=tk.X, pady=2)

        
        tk.Label(panel, text="Status:", font=("Arial", 10, "bold"), bg=bg_color).pack(anchor="w", pady=(20, 0))
        self.lbl_status = tk.Label(panel, text="Ready.", bg="#e8e8e8", fg="#333",
                                   justify="left", wraplength=300, relief="sunken", height=6)
        self.lbl_status.pack(fill=tk.X, pady=5)

        
        self.canvas = tk.Canvas(self.root, bg="#2b2b2b", cursor="cross")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

    

    def load_image(self):
        











        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder:
            return

        selected_dir = Path(folder)
        root_dir = resolve_image_folder_from_selected_folder(selected_dir)
        self.folder_path = str(root_dir)
        self.load_metadata_for_folder(root_dir)

        
        imgs = [
            p for p in root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in FOLDER_IMAGE_SUFFIXES
        ]
        imgs = sorted(imgs, key=lambda p: str(p).lower())

        if not imgs:
            messagebox.showerror("Error", "No supported images found in the selected folder.")
            return

        self.image_list = imgs
        self.image_index = 0
        self.load_current_image_from_folder()

        
        self.btn_sel_hor.config(state="normal")
        self.btn_sel_ver.config(state="normal")
        self.btn_reset_calib.config(state="normal")
        self.btn_measure.config(state="normal")
        self.btn_clear_meas.config(state="normal")

    def load_metadata_for_folder(self, image_folder):
        




        self.metadata_lookup = {}
        self.metadata_path = find_metadata_csv_for_image_folder(Path(image_folder))

        if self.metadata_path is None:
            return

        try:
            df = pd.read_csv(self.metadata_path)
        except Exception:
            self.metadata_path = None
            return

        if "image_name" not in df.columns:
            self.metadata_path = None
            return

        for _, row in df.iterrows():
            names = []
            image_name = str(row.get("image_name", "")).strip()
            if image_name:
                names.append(image_name)
            image_path = str(row.get("image_path", "")).strip()
            if image_path:
                names.append(Path(image_path).name)

            for name in names:
                if name:
                    self.metadata_lookup[name] = row

    def parse_heights_from_image_path(self, path):
        








        p = Path(path)
        name = p.name

        
        row = self.metadata_lookup.get(name)
        if row is not None:
            try:
                cam = float(row.get("camera_height_cm"))
                true_h = float(row.get("true_height_cm"))
                if np.isfinite(cam) and np.isfinite(true_h):
                    return cam, true_h, "metadata_csv"
            except Exception:
                pass

        
        m = re.match(r"^(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)__", name)
        if m:
            return float(m.group(1)), float(m.group(2)), "filename_prefix"

        
        parts = p.parts
        for i in range(len(parts) - 2):
            try:
                cam = float(parts[i])
                true_h = float(parts[i + 1])
                return cam, true_h, "parent_folders"
            except Exception:
                pass

        return None, None, "not_found"

    def load_current_image_from_folder(self):
        if not self.image_list:
            return

        if self.image_index < 0:
            self.image_index = 0
        if self.image_index >= len(self.image_list):
            self.image_index = len(self.image_list) - 1

        path = self.image_list[self.image_index]
        self.img_path = str(path)
        self.pil_image = Image.open(path).convert("RGB")
        self.tk_image = None
        self.img_w, self.img_h = self.pil_image.size

        
        self.principal_point = (self.img_w / 2, self.img_h / 2)

        cam_cm, true_cm, parse_source = self.parse_heights_from_image_path(path)
        self.current_camera_height_cm = cam_cm
        self.current_true_height_cm = true_cm
        self.current_height_parse_source = parse_source

        
        if cam_cm is not None:
            self.entry_hc.delete(0, tk.END)
            self.entry_hc.insert(0, f"{cam_cm / 100.0:.3f}")

        self.reset_all_data()
        self.resize_image()

        msg = (
            f"Image Loaded [{self.image_index + 1}/{len(self.image_list)}].\n"
            f"{path.name}\n"
            f"Res: {self.img_w}x{self.img_h}\n"
            f"Center Cy={self.principal_point[1]:.1f}"
        )
        if cam_cm is not None and true_cm is not None:
            msg += f"\nHc={cam_cm:.1f}cm, True={true_cm:.1f}cm"
            msg += f"\nHeight source={parse_source}"
        else:
            msg += "\nHeight info not found in metadata/filename."
        if self.metadata_path is not None:
            msg += f"\nMetadata={self.metadata_path.name}"
        msg += "\nNext: Right/Space/N, Prev: Left/P, Run: Enter"

        self.lbl_status.config(text=msg)

        
        try:
            self.root.focus_force()
            self.canvas.focus_set()
        except Exception:
            pass

    def next_image(self):
        if not self.image_list:
            return
        if self.image_index >= len(self.image_list) - 1:
            self.lbl_status.config(text="Already at last image.")
            return
        self.image_index += 1
        self.load_current_image_from_folder()

    def prev_image(self):
        if not self.image_list:
            return
        if self.image_index <= 0:
            self.lbl_status.config(text="Already at first image.")
            return
        self.image_index -= 1
        self.load_current_image_from_folder()


    def quick_run_current_image(self):
        


        if not self.image_list or self.pil_image is None:
            return
        if self.mode_var.get() == "manual":
            return
        self.do_auto_measure()


    def resize_image(self, event=None):
        if not self.pil_image: return

        
        can_w = self.canvas.winfo_width()
        can_h = self.canvas.winfo_height()

        if can_w < 10 or can_h < 10: return

        
        ratio = min(can_w / self.img_w, can_h / self.img_h)
        self.scale = ratio
        new_w, new_h = int(self.img_w * ratio), int(self.img_h * ratio)

        
        if hasattr(self, 'tk_image') and self.tk_image:
            if self.tk_image.width() == new_w and self.tk_image.height() == new_h:
                self.redraw()
                return

        
        resized = self.pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        
        self.offset_x = (can_w - new_w) // 2
        self.offset_y = (can_h - new_h) // 2

        self.redraw()

    def reset_all_data(self):
        self.lines_horizon = []
        self.lines_vertical = []
        self.focal_length = 0
        self.horizon_y = 0
        self.vz_y = 0
        self.clicks = []
        self.auto_detection = None
        self.auto_priors = None
        self.auto_candidate = None
        self.auto_height_result = None
        self._computed_auto_result = None
        self.canvas.delete("all")
        self.redraw()

    def reset_calibration_selection(self):
        
        self.lines_horizon = []
        self.lines_vertical = []
        self.clicks = []
        self.state = "IDLE"
        self.auto_priors = None
        self.auto_candidate = None
        self.canvas.delete("user_lines")
        self.canvas.delete("temp")
        self.canvas.delete("calc_horizon")
        self.canvas.delete("auto_lines")
        self.canvas.delete("stage_res")
        self.lbl_status.config(text="Calibration selections cleared.\nPlease re-select lines.")

    def clear_measurements(self):
        
        self.canvas.delete("measure_res")
        self.canvas.delete("stage_res")
        self.auto_height_result = None
        self.lbl_status.config(text="Measurements cleared.")

    def toggle_mode(self):
        is_manual = self.mode_var.get() == "manual"
        state = "disabled" if is_manual else "normal"
        self.btn_sel_hor.config(state=state)
        self.btn_sel_ver.config(state=state)
        self.btn_reset_calib.config(state=state)
        self.ent_f.config(state="normal" if is_manual else "disabled")
        self.ent_pitch.config(state="normal" if is_manual else "disabled")

    def select_horizon_button_action(self):
        




        if self.mode_var.get() == "manual":
            self.set_state("GET_HORIZON")
            return

        try:
            result = self.ensure_auto_result_for_stage()
            self.draw_horizon_stage_result(result)
        except Exception as e:
            messagebox.showerror("Auto Horizon Error", repr(e))
            self.lbl_status.config(text=f"Auto horizon failed:\n{repr(e)}")

    def select_vertical_button_action(self):
        



        if self.mode_var.get() == "manual":
            self.set_state("GET_VERTICAL")
            return

        try:
            result = self.ensure_auto_result_for_stage()
            self.draw_vertical_stage_result(result)
        except Exception as e:
            messagebox.showerror("Auto Vertical Error", repr(e))
            self.lbl_status.config(text=f"Auto vertical failed:\n{repr(e)}")

    def ensure_auto_result_for_stage(self):
        



        if self.pil_image is None:
            raise RuntimeError("Please open an image folder/image first.")

        if self._computed_auto_result is not None:
            return self._computed_auto_result

        result = self.run_exact_method4_on_current_image()

        self.auto_detection = result["det"]
        self.auto_priors = result["priors"]
        self.auto_candidate = result["main_candidate"]
        self._computed_auto_result = result

        
        self.auto_height_result = None
        return result

    def set_state(self, new_state):
        self.state = new_state
        self.clicks = []
        self.canvas.delete("temp")  

        msg = {
            "GET_HORIZON": "Draw 2 [HORIZONTAL] Parallel Lines.\n(e.g., floor tiles, beams)\nClick: Line1 Start->End, Line2 Start->End",
            "GET_VERTICAL": "Draw 2 [VERTICAL] Parallel Lines.\n(e.g., walls, pillars)\nClick: Line1 Start->End, Line2 Start->End",
            "MEASURE": "Measure Object:\n1. Click Top (Head)\n2. Click Bottom (Feet/Ground)"
        }
        self.lbl_status.config(text=msg.get(new_state, ""))

    def measure_button_action(self):
        if self.mode_var.get() == "manual":
            self.set_state("MEASURE")
        else:
            self.do_auto_measure()

    def on_click(self, event):
        if self.state == "IDLE": return

        
        img_x = (event.x - self.offset_x) / self.scale
        img_y = (event.y - self.offset_y) / self.scale

        self.clicks.append((img_x, img_y))

        
        cx, cy = event.x, event.y
        self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="red", outline="white", tags="temp")

        
        if self.state in ["GET_HORIZON", "GET_VERTICAL"]:
            if len(self.clicks) == 2:
                
                p1 = self.to_canvas(self.clicks[0])
                p2 = self.to_canvas(self.clicks[1])
                color = "cyan" if "HORIZON" in self.state else "lime"
                self.canvas.create_line(p1, p2, fill=color, width=2, tags="user_lines")

            if len(self.clicks) == 4:
                
                p3 = self.to_canvas(self.clicks[2])
                p4 = self.to_canvas(self.clicks[3])
                color = "cyan" if "HORIZON" in self.state else "lime"
                self.canvas.create_line(p3, p4, fill=color, width=2, tags="user_lines")

                line_data = [(self.clicks[0], self.clicks[1]), (self.clicks[2], self.clicks[3])]

                if "HORIZON" in self.state:
                    self.lines_horizon = line_data
                    self.lbl_status.config(text="Horizon references saved.")
                else:
                    self.lines_vertical = line_data
                    self.lbl_status.config(text="Vertical references saved.")

                self.state = "IDLE"
                self.clicks = []
                self.canvas.delete("temp")

        
        elif self.state == "MEASURE":
            if len(self.clicks) == 2:
                self.do_measure(self.clicks[0], self.clicks[1])
                self.state = "IDLE"
                self.clicks = []
                self.canvas.delete("temp")

    

    def get_line_intersection_homogeneous(self, line1_pts, line2_pts):
        
        p1 = np.array([line1_pts[0][0], line1_pts[0][1], 1.0])
        p2 = np.array([line1_pts[1][0], line1_pts[1][1], 1.0])
        p3 = np.array([line2_pts[0][0], line2_pts[0][1], 1.0])
        p4 = np.array([line2_pts[1][0], line2_pts[1][1], 1.0])

        l1 = np.cross(p1, p2)
        l2 = np.cross(p3, p4)

        v = np.cross(l1, l2)

        if abs(v[2]) < 1e-5:
            return None  

        return (v[0] / v[2], v[1] / v[2])

    def calculate_camera_params(self):
        if self.mode_var.get() == "manual":
            messagebox.showinfo("Info", "Manual mode selected. No calculation needed.")
            return

        
        try:
            self.calculate_camera_params_auto()
        except Exception as e:
            messagebox.showerror("Auto Calibration Error", repr(e))
            self.lbl_status.config(text=f"Auto calibration failed:\n{repr(e)}")

    
    
    

    def get_cv_image_bgr(self):
        if self.pil_image is None:
            return None
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed. Please install: pip install opencv-python")
        arr = np.array(self.pil_image.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def ensure_yolo_model(self):
        if YOLO is None:
            raise RuntimeError("ultralytics is not installed. Please install: pip install ultralytics")

        if self.yolo_model is None:
            model_path = resolve_yolo_model_path()
            self.yolo_model = YOLO(str(model_path))

    def ensure_geom_module(self):
        




        if hasattr(self, "geom_module") and self.geom_module is not None:
            return self.geom_module

        geom_path = resolve_geom_method_script_path()

        spec = importlib.util.spec_from_file_location("geom_method4_exact", str(geom_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to import geom method script: {geom_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.geom_module = module
        return module

    def get_keypoint_arrays_from_result(self, result):
        





        if result.keypoints is None:
            return None, None

        kpts_xy = result.keypoints.xy
        if kpts_xy is None:
            return None, None

        kpts_xy = kpts_xy.cpu().numpy()

        kpts_conf = None
        if hasattr(result.keypoints, "conf") and result.keypoints.conf is not None:
            kpts_conf = result.keypoints.conf.cpu().numpy()

        return kpts_xy, kpts_conf

    def choose_best_detection_exact(self, result, image_w, image_h):
        




        if result.boxes is None or len(result.boxes) == 0:
            return None

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()

        kpts_xy, kpts_conf = self.get_keypoint_arrays_from_result(result)

        img_center_x = image_w / 2.0
        img_center_y = image_h / 2.0
        max_dist = np.sqrt(img_center_x ** 2 + img_center_y ** 2)

        best_idx = None
        best_score = -1.0

        for i in range(len(boxes_conf)):
            box_conf = float(boxes_conf[i])

            if kpts_conf is not None and i < len(kpts_conf):
                kp_conf_mean = float(np.nanmean(kpts_conf[i]))
            else:
                kp_conf_mean = box_conf

            geometry_quality = 0.0

            if kpts_xy is not None and i < len(kpts_xy) and len(kpts_xy[i]) >= 2:
                top_x, top_y = kpts_xy[i][0]
                base_x, base_y = kpts_xy[i][1]

                height_in_image = abs(float(base_y) - float(top_y))

                if height_in_image < 5:
                    geometry_quality = 0.2
                else:
                    geometry_quality = 1.0

                if float(top_y) >= float(base_y):
                    geometry_quality *= 0.3

            if USE_CENTER_PRIORITY:
                x1, y1, x2, y2 = boxes_xyxy[i]
                box_cx = (x1 + x2) / 2.0
                box_cy = (y1 + y2) / 2.0
                dist = np.sqrt((box_cx - img_center_x) ** 2 + (box_cy - img_center_y) ** 2)
                center_score = 1.0 - min(dist / max_dist, 1.0)

                score = (
                    0.60 * box_conf
                    + 0.25 * kp_conf_mean
                    + 0.10 * center_score
                    + 0.05 * geometry_quality
                )
            else:
                score = (
                    0.70 * box_conf
                    + 0.25 * kp_conf_mean
                    + 0.05 * geometry_quality
                )

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            return None

        return {
            "index": best_idx,
            "score": float(best_score),
            "box_xyxy": boxes_xyxy[best_idx],
            "box_conf": float(boxes_conf[best_idx]),
            "kpts_xy": None if kpts_xy is None else kpts_xy[best_idx],
            "kpts_conf": None if kpts_conf is None else kpts_conf[best_idx],
        }

    def run_yolo_detection(self):
        









        if self.img_path is None:
            raise RuntimeError("Please open an image first.")

        if self.auto_detection is not None:
            return self.auto_detection

        self.ensure_yolo_model()

        
        image = cv2.imread(str(self.img_path))
        if image is None:
            raise RuntimeError(f"Failed to read image by cv2.imread: {self.img_path}")
        image_h, image_w = image.shape[:2]

        try:
            results = self.yolo_model.predict(
                source=self.img_path,
                task="pose",
                imgsz=IMGSZ,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                max_det=MAX_DET_CANDIDATES,
                device=YOLO_DEVICE,
                verbose=False,
            )
        except Exception as e:
            
            
            results = self.yolo_model.predict(
                source=self.img_path,
                task="pose",
                imgsz=IMGSZ,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                max_det=MAX_DET_CANDIDATES,
                device="cpu",
                verbose=False,
            )

        if not results:
            raise RuntimeError("YOLO returned no result.")

        r = results[0]
        best = self.choose_best_detection_exact(r, image_w=image_w, image_h=image_h)

        if best is None:
            raise RuntimeError("YOLO no_detection: choose_best_detection returned None.")

        kpts = best["kpts_xy"]
        kpts_conf = best["kpts_conf"]

        if kpts is None or len(kpts) < 2:
            raise RuntimeError("YOLO keypoints missing or less than 2.")

        
        
        
        top = tuple(float(v) for v in kpts[KEYPOINT_INDEX_TOP][:2])
        base = tuple(float(v) for v in kpts[KEYPOINT_INDEX_BASE][:2])
        bbox = tuple(float(v) for v in best["box_xyxy"])

        
        
        if not np.all(np.isfinite([top[0], top[1], base[0], base[1]])):
            raise RuntimeError("YOLO keypoints contain NaN or invalid values.")

        top_conf = ""
        base_conf = ""
        if kpts_conf is not None and len(kpts_conf) >= 2:
            top_conf = float(kpts_conf[KEYPOINT_INDEX_TOP])
            base_conf = float(kpts_conf[KEYPOINT_INDEX_BASE])

        self.auto_detection = {
            "top": top,
            "base": base,
            "bbox": bbox,
            "conf": float(best["box_conf"]),
            "selected_score": float(best["score"]),
            "top_conf": top_conf,
            "base_conf": base_conf,
            "best_index": int(best["index"]),
        }

        return self.auto_detection

    def make_method4_row_from_current_image(self, det):
        



        top_x, top_y = det["top"]
        base_x, base_y = det["base"]
        x1, y1, x2, y2 = det["bbox"]

        
        H_cam_cm = self.current_camera_height_cm
        true_h_cm = self.current_true_height_cm

        if H_cam_cm is None or not np.isfinite(H_cam_cm):
            try:
                H_cam_cm = float(self.entry_hc.get()) * 100.0
            except Exception:
                H_cam_cm = np.nan

        if true_h_cm is None or not np.isfinite(true_h_cm):
            true_h_cm = np.nan

        row = {
            "image_path": str(self.img_path),
            "relative_path": Path(self.img_path).name if self.img_path else "",
            "image_name": Path(self.img_path).name if self.img_path else "",
            "status": "ok",
            "camera_setting": float(H_cam_cm) if np.isfinite(H_cam_cm) else np.nan,
            "true_height_cm": float(true_h_cm) if np.isfinite(true_h_cm) else np.nan,
            "bbox_x1": float(x1),
            "bbox_y1": float(y1),
            "bbox_x2": float(x2),
            "bbox_y2": float(y2),
            "top_x": float(top_x),
            "top_y": float(top_y),
            "base_x": float(base_x),
            "base_y": float(base_y),
            "selected_score": float(det.get("selected_score", det.get("conf", np.nan))),
            "box_conf": float(det.get("conf", np.nan)),
            "top_conf": det.get("top_conf", ""),
            "base_conf": det.get("base_conf", ""),
        }
        return pd.Series(row)

    def run_exact_method4_on_current_image(self):
        









        geom = self.ensure_geom_module()
        det = self.run_yolo_detection()
        row = self.make_method4_row_from_current_image(det)

        
        img = geom.imread_unicode(Path(self.img_path))
        if img is None:
            raise RuntimeError(f"image_read_failed: {self.img_path}")

        img_h, _ = img.shape[:2]

        
        kp_ok, kp_status = geom.keypoints_are_valid(row, img_h)
        if not kp_ok:
            raise RuntimeError(f"keypoints invalid by method4: {kp_status}")

        H_cam_cm = geom.safe_float(row.get("camera_setting"))
        true_height_cm = geom.safe_float(row.get("true_height_cm"))

        top_y = geom.safe_float(row.get("top_y"))
        base_y = geom.safe_float(row.get("base_y"))

        priors = geom.estimate_geometric_priors(img, row)

        candidate_eval = {}
        for source, candidate in priors["candidates"].items():
            ev = geom.evaluate_candidate_height(
                candidate=candidate,
                H_cam_cm=H_cam_cm,
                true_height_cm=true_height_cm,
                y_top=top_y,
                y_bot=base_y,
                y_v=priors["y_v"],
                vertical_vp_reliable=priors["vertical_vp_reliable"],
                img_h=img_h,
            )
            candidate_eval[source] = ev

        main_candidate, select_reason = geom.select_main_candidate(priors, candidate_eval, img_h)

        if main_candidate is None:
            raise RuntimeError("No valid horizon candidate by exact method4. " + select_reason)

        main_source = main_candidate["source"]
        main_eval = candidate_eval.get(main_source, {})
        est_h_cm = geom.safe_float(main_eval.get(f"{main_source}_estimated_height_cm"))
        height_mode = main_eval.get(f"{main_source}_height_mode", "")
        height_status = main_eval.get(f"{main_source}_height_status", "")

        if not geom.is_plausible_height(est_h_cm):
            raise RuntimeError(f"Invalid height result by exact method4: {est_h_cm}; {height_status}")

        abs_error_cm = np.nan
        rel_error_percent = np.nan
        if np.isfinite(true_height_cm) and true_height_cm > 0:
            abs_error_cm = abs(est_h_cm - true_height_cm)
            rel_error_percent = abs_error_cm / true_height_cm * 100.0

        result = {
            "det": det,
            "row": row,
            "img_bgr": img,
            "priors": priors,
            "candidate_eval": candidate_eval,
            "main_candidate": main_candidate,
            "selected_horizon_source": main_source,
            "select_reason": select_reason,
            "estimated_height_cm": float(est_h_cm),
            "estimated_height_m": float(est_h_cm) / 100.0,
            "height_mode": height_mode,
            "height_status": height_status,
            "true_height_cm": float(true_height_cm) if np.isfinite(true_height_cm) else np.nan,
            "abs_error_cm": float(abs_error_cm) if np.isfinite(abs_error_cm) else np.nan,
            "rel_error_percent": float(rel_error_percent) if np.isfinite(rel_error_percent) else np.nan,
            "y_hor_px": float(main_candidate.get("y_hor", np.nan)),
            "y_v_px": float(priors.get("y_v", np.nan)) if np.isfinite(priors.get("y_v", np.nan)) else np.nan,
            "vertical_vp_reliable": bool(priors.get("vertical_vp_reliable", False)),
        }
        return result

    def calculate_camera_params_auto(self):
        if self.pil_image is None:
            raise RuntimeError("Please open an image first.")

        result = self.run_exact_method4_on_current_image()

        self.auto_detection = result["det"]
        self.auto_priors = result["priors"]
        self.auto_candidate = result["main_candidate"]
        self._computed_auto_result = result
        
        self.auto_height_result = None

        self.horizon_y = result["y_hor_px"]
        self.vz_y = result["y_v_px"]

        
        cx, cy = self.principal_point
        dy_hor = self.horizon_y - cy if np.isfinite(self.horizon_y) else 0.0

        if result["vertical_vp_reliable"] and np.isfinite(self.vz_y):
            dy_ver = self.vz_y - cy
            product = -(dy_hor * dy_ver)
            if product > 0:
                self.focal_length = math.sqrt(product)
                self.pitch = math.atan(dy_hor / self.focal_length)
            else:
                self.focal_length = float(self.img_w)
                self.pitch = math.atan(dy_hor / self.focal_length)
        else:
            self.focal_length = float(self.img_w)
            self.pitch = math.atan(dy_hor / self.focal_length)

        self.ent_f.delete(0, tk.END)
        if result["vertical_vp_reliable"]:
            self.ent_f.insert(0, f"{self.focal_length:.1f}")
        else:
            self.ent_f.insert(0, f"{self.focal_length:.1f} (Dummy)")

        self.ent_pitch.delete(0, tk.END)
        self.ent_pitch.insert(0, f"{math.degrees(self.pitch):.2f}")

        
        self.canvas.delete("calc_horizon")
        self.canvas.delete("auto_lines")
        can_y = self.offset_y + self.horizon_y * self.scale
        self.canvas.create_line(0, can_y, 3000, can_y, fill="cyan", dash=(4, 4), width=2, tags="calc_horizon")
        self.canvas.create_text(10, can_y - 10, text="Calculated Horizon", fill="cyan", anchor="w",
                                tags="calc_horizon")

        self.draw_auto_hough_lines(result["priors"])
        self.draw_yolo_detection(result["det"])

        status = (
            f"Success!\n"
            f"Focal Length f={self.focal_length:.1f} px\n"
            f"Pitch Angle={math.degrees(self.pitch):.2f}°\n"
            f"Horizon Y={self.horizon_y:.1f}\n"
            f"Auto={result['selected_horizon_source']}"
        )
        self.lbl_status.config(text=status)

    def do_auto_measure(self):
        if self.pil_image is None:
            messagebox.showwarning("Warning", "Please open an image first.")
            return

        try:
            result = self.run_exact_method4_on_current_image()
            self.auto_detection = result["det"]
            self.auto_priors = result["priors"]
            self.auto_candidate = result["main_candidate"]
            self.auto_height_result = result
            self._computed_auto_result = result
            self.horizon_y = result["y_hor_px"]
            self.vz_y = result["y_v_px"]

            self.draw_measurement_result(
                result["det"]["top"],
                result["det"]["base"],
                result["estimated_height_m"],
                result,
                result["main_candidate"],
            )

        except Exception as e:
            messagebox.showerror("Auto Measurement Error", repr(e))
            self.lbl_status.config(text=f"Auto measurement failed:\n{repr(e)}")

    

    def do_measure(self, p_top, p_bot):
        
        try:
            Hc = float(self.entry_hc.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid Camera Height")
            return

        
        if self.mode_var.get() == "manual":
            try:
                f = float(self.ent_f.get())
                deg = float(self.ent_pitch.get())
                pitch = math.radians(deg)
                cy = self.principal_point[1]
                
                horizon_y = f * math.tan(pitch) + cy
            except ValueError:
                messagebox.showerror("Error", "Invalid manual parameters")
                return
        else:
            f = self.focal_length
            pitch = self.pitch
            cy = self.principal_point[1]
            horizon_y = self.horizon_y
            if f == 0:
                messagebox.showwarning("Warning", "Please calculate camera parameters first.")
                return

        
        
        y_top = min(p_top[1], p_bot[1])
        y_bot = max(p_top[1], p_bot[1])

        
        alpha_top = math.atan((y_top - cy) / f)
        alpha_bot = math.atan((y_bot - cy) / f)

        
        theta = math.atan((horizon_y - cy) / f)

        
        dip_top = alpha_top - theta
        dip_bot = alpha_bot - theta

        
        if abs(math.tan(dip_bot)) < 1e-4:
            res = 0  
        else:
            res = Hc * (1 - math.tan(dip_top) / math.tan(dip_bot))

        
        p1_can = self.to_canvas(p_top)
        p2_can = self.to_canvas(p_bot)

        
        self.canvas.create_line(p1_can, p2_can, fill="yellow", width=3, arrow=tk.BOTH, tags="measure_res")

        
        text_x = p1_can[0] + 15
        text_y = (p1_can[1] + p2_can[1]) / 2

        
        
        display_text = f"{abs(res):.3f}m"

        
        self.canvas.create_rectangle(text_x - 2, text_y - 10, text_x + 100, text_y + 10,
                                     fill="#333", outline="yellow", tags="measure_res")

        self.canvas.create_text(text_x, text_y, text=display_text, fill="yellow",
                                font=("Arial", 12, "bold"), anchor="w", tags="measure_res")
        

        self.lbl_status.config(
            text=f"Measurement: {abs(res):.3f} meters\n(DipTop={math.degrees(dip_top):.1f}°, DipBot={math.degrees(dip_bot):.1f}°)")

    
    def draw_segment_list(self, segs, color, width=2, limit=120, tags="stage_res"):
        





        if segs is None:
            return
        segs_sorted = sorted(segs, key=lambda s: s["length"], reverse=True)[:limit]
        for s in segs_sorted:
            p1 = self.to_canvas((s["x1"], s["y1"]))
            p2 = self.to_canvas((s["x2"], s["y2"]))
            self.canvas.create_line(p1, p2, fill=color, width=width, tags=tags)

    def select_representative_segments(self, segs, count=2):
        








        if not segs:
            return []

        
        candidates = sorted(segs, key=lambda s: float(s.get("length", 0.0)), reverse=True)

        selected = []
        min_sep = 0.18 * max(float(self.img_w), float(self.img_h), 1.0)

        def midpoint(seg):
            return (
                0.5 * (float(seg["x1"]) + float(seg["x2"])),
                0.5 * (float(seg["y1"]) + float(seg["y2"])),
            )

        for seg in candidates:
            if len(selected) >= count:
                break
            if not selected:
                selected.append(seg)
                continue

            mx, my = midpoint(seg)
            far_enough = True
            for s0 in selected:
                m0x, m0y = midpoint(s0)
                if math.hypot(mx - m0x, my - m0y) < min_sep:
                    far_enough = False
                    break
            if far_enough:
                selected.append(seg)

        
        if len(selected) < count:
            for seg in candidates:
                if len(selected) >= count:
                    break
                if seg not in selected:
                    selected.append(seg)

        return selected[:count]

    def draw_representative_segment_list(self, segs, color, width=3, count=2, tags="stage_res"):
        
        for s in self.select_representative_segments(segs, count=count):
            p1 = self.to_canvas((s["x1"], s["y1"]))
            p2 = self.to_canvas((s["x2"], s["y2"]))
            self.canvas.create_line(p1, p2, fill=color, width=width, tags=tags)

    def draw_point_label(self, pt, label, color, tags="stage_res"):
        


        if pt is None:
            return
        try:
            x, y = float(pt[0]), float(pt[1])
        except Exception:
            return

        margin = 0.5 * max(self.img_w, self.img_h)
        if not (-margin <= x <= self.img_w + margin and -margin <= y <= self.img_h + margin):
            return

        cx, cy = self.to_canvas((x, y))
        self.canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6, fill=color, outline="white", tags=tags)
        self.canvas.create_text(cx + 10, cy - 10, text=label, fill=color,
                                font=("Arial", 10, "bold"), anchor="w", tags=tags)

    def draw_stage_title(self, text, color="white"):
        self.canvas.create_rectangle(12, 12, 470, 76, fill="#333333", outline=color, tags="stage_res")
        self.canvas.create_text(24, 24, text=text, fill=color,
                                font=("Arial", 12, "bold"), anchor="nw", tags="stage_res")

    def draw_horizon_stage_result(self, result):
        



        self.canvas.delete("stage_res")
        self.canvas.delete("auto_lines")
        self.canvas.delete("calc_horizon")
        self.canvas.delete("measure_res")

        priors = result["priors"]
        groups = priors.get("groups", {})
        candidates = priors.get("candidates", {})
        main_candidate = result["main_candidate"]
        self.auto_candidate = main_candidate

        
        
        self.draw_representative_segment_list(groups.get("horizontal", []), "magenta", width=3, count=2, tags="stage_res")

        
        self.draw_infinite_line(main_candidate.get("line"), "yellow", 5, tags="stage_res")

        self.draw_stage_title(
            "Two representative horizon cues shown\nyellow=selected horizon",
            color="yellow"
        )

        idx_text = f" [{self.image_index + 1}/{len(self.image_list)}]" if self.image_list else ""
        self.lbl_status.config(
            text=f"Horizon result{idx_text}\n"
                 f"Selected={result['selected_horizon_source']}\n"
                 f"Y_horizon={result['y_hor_px']:.1f}px\n"
                 f"Next: click Select Vertical"
        )

    def draw_vertical_stage_result(self, result):
        



        self.canvas.delete("stage_res")
        self.canvas.delete("auto_lines")
        self.canvas.delete("calc_horizon")
        self.canvas.delete("measure_res")

        priors = result["priors"]
        groups = priors.get("groups", {})

        
        
        self.draw_representative_segment_list(groups.get("vertical", []), "lime", width=3, count=2, tags="stage_res")

        
        self.draw_point_label(priors.get("vp_vertical"), "VVP", "lime", tags="stage_res")

        if result["vertical_vp_reliable"]:
            vvp_text = f"Vertical VP used, y_v={result['y_v_px']:.1f}px"
        else:
            vvp_text = "Vertical VP unreliable; use yv-infinity fallback"

        self.draw_stage_title(
            "Two representative vertical cues shown\nVVP label if visible",
            color="lime"
        )

        idx_text = f" [{self.image_index + 1}/{len(self.image_list)}]" if self.image_list else ""
        self.lbl_status.config(
            text=f"Vertical result{idx_text}\n"
                 f"{vvp_text}\n"
                 f"Next: click Calculate Focal & Pitch"
        )

    def draw_infinite_line(self, line, color, thickness=3, tags="auto_lines"):
        if line is None:
            return

        h, w = self.img_h, self.img_w
        a, b, c = line
        pts = []

        if abs(b) > 1e-9:
            pts.append((0, int(round(-(a * 0 + c) / b))))
            pts.append((w - 1, int(round(-(a * (w - 1) + c) / b))))

        if abs(a) > 1e-9:
            pts.append((int(round(-(b * 0 + c) / a)), 0))
            pts.append((int(round(-(b * (h - 1) + c) / a)), h - 1))

        valid = []
        for x, y in pts:
            if -w <= x <= 2 * w and -h <= y <= 2 * h:
                valid.append((x, y))

        if len(valid) >= 2:
            p1 = self.to_canvas(valid[0])
            p2 = self.to_canvas(valid[1])
            self.canvas.create_line(p1, p2, fill=color, width=thickness, tags=tags)

    def draw_auto_hough_lines(self, priors):
        self.canvas.delete("auto_lines")

        
        
        groups = priors.get("groups", {})
        self.draw_representative_segment_list(groups.get("horizontal", []), "magenta", width=2, count=2, tags="auto_lines")

        if self.auto_candidate is not None:
            self.draw_infinite_line(self.auto_candidate.get("line"), "yellow", 4)

    def draw_yolo_detection(self, det):
        
        x1, y1, x2, y2 = det["bbox"]
        p1 = self.to_canvas((x1, y1))
        p2 = self.to_canvas((x2, y2))
        self.canvas.create_rectangle(p1[0], p1[1], p2[0], p2[1], outline="orange", width=2, tags="measure_res")

        top = det["top"]
        base = det["base"]
        top_can = self.to_canvas(top)
        base_can = self.to_canvas(base)

        self.canvas.create_oval(top_can[0] - 6, top_can[1] - 6, top_can[0] + 6, top_can[1] + 6,
                                fill="red", outline="white", tags="measure_res")
        self.canvas.create_text(top_can[0] + 10, top_can[1] - 10, text="top",
                                fill="red", font=("Arial", 10, "bold"), anchor="w", tags="measure_res")

        self.canvas.create_oval(base_can[0] - 6, base_can[1] - 6, base_can[0] + 6, base_can[1] + 6,
                                fill="blue", outline="white", tags="measure_res")
        self.canvas.create_text(base_can[0] + 10, base_can[1] - 10, text="base",
                                fill="blue", font=("Arial", 10, "bold"), anchor="w", tags="measure_res")

        self.canvas.create_line(top_can, base_can, fill="white", width=2, tags="measure_res")

    def draw_measurement_result(self, top, base, height_m, ev, candidate):
        self.canvas.delete("measure_res")
        self.canvas.delete("auto_lines")

        if self.auto_priors is not None:
            self.draw_auto_hough_lines(self.auto_priors)

        self.draw_yolo_detection(self.auto_detection)

        
        self.draw_infinite_line(candidate.get("line"), "yellow", 4)

        p1_can = self.to_canvas(top)
        p2_can = self.to_canvas(base)

        
        self.canvas.create_line(p1_can, p2_can, fill="yellow", width=3, arrow=tk.BOTH, tags="measure_res")

        text_x = p1_can[0] + 15
        text_y = (p1_can[1] + p2_can[1]) / 2
        display_text = f"{abs(height_m):.3f}m"

        self.canvas.create_rectangle(text_x - 2, text_y - 10, text_x + 100, text_y + 10,
                                     fill="#333", outline="yellow", tags="measure_res")

        self.canvas.create_text(text_x, text_y, text=display_text, fill="yellow",
                                font=("Arial", 12, "bold"), anchor="w", tags="measure_res")

        extra = ""
        if self.current_true_height_cm is not None:
            true_m = self.current_true_height_cm / 100.0
            abs_err_m = abs(abs(height_m) - true_m)
            rel_err = abs_err_m / true_m * 100.0 if true_m > 0 else float("nan")
            extra = f"\nTrue={true_m:.3f}m, RE={rel_err:.2f}%"

        idx_text = ""
        if self.image_list:
            idx_text = f" [{self.image_index + 1}/{len(self.image_list)}]"

        self.lbl_status.config(
            text=f"Measurement{idx_text}: {abs(height_m):.3f} meters\n"
                 f"Mode={ev['height_mode']}\n"
                 f"Horizon={candidate.get('source', '')}\n"
                 f"Conf={self.auto_detection.get('conf', 0):.3f}"
                 f"{extra}\nNext: Right/Space/N, Prev: Left/P, Run: Enter"
        )

    
    def redraw(self):
        self.canvas.delete("img_bg")  
        self.canvas.delete("center_marker")

        if self.tk_image:
            self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image, tags="img_bg")
            self.canvas.tag_lower("img_bg")  

        
        cx, cy = self.principal_point
        cx_can = self.offset_x + cx * self.scale
        cy_can = self.offset_y + cy * self.scale
        self.canvas.create_line(cx_can - 10, cy_can, cx_can + 10, cy_can, fill="red", tags="center_marker")
        self.canvas.create_line(cx_can, cy_can - 10, cx_can, cy_can + 10, fill="red", tags="center_marker")

        
        if self.horizon_y != 0:
            can_y = self.offset_y + self.horizon_y * self.scale
            self.canvas.delete("calc_horizon")
            self.canvas.create_line(0, can_y, 3000, can_y, fill="cyan", dash=(4, 4), width=2, tags="calc_horizon")
            self.canvas.create_text(10, can_y - 10, text="Calculated Horizon", fill="cyan", anchor="w",
                                    tags="calc_horizon")

        
        
        
        
        self.redraw_user_lines()

        
        if self.auto_priors is not None:
            self.draw_auto_hough_lines(self.auto_priors)

        if self.auto_detection is not None and self.auto_height_result is None:
            self.canvas.delete("measure_res")
            self.draw_yolo_detection(self.auto_detection)

        if self.auto_detection is not None and self.auto_height_result is not None and self.auto_candidate is not None:
            self.draw_measurement_result(
                self.auto_detection["top"],
                self.auto_detection["base"],
                self.auto_height_result["estimated_height_m"],
                self.auto_height_result,
                self.auto_candidate,
            )

    def redraw_user_lines(self):
        
        self.canvas.delete("user_lines")

        
        for line in self.lines_horizon:
            p1 = self.to_canvas(line[0])
            p2 = self.to_canvas(line[1])
            self.canvas.create_line(p1, p2, fill="cyan", width=2, tags="user_lines")

        
        for line in self.lines_vertical:
            p1 = self.to_canvas(line[0])
            p2 = self.to_canvas(line[1])
            self.canvas.create_line(p1, p2, fill="lime", width=2, tags="user_lines")

    def to_canvas(self, pt):
        return (self.offset_x + pt[0] * self.scale, self.offset_y + pt[1] * self.scale)


if __name__ == "__main__":
    root = tk.Tk()
    app = RobustMetrologyApp(root)
    root.mainloop()
