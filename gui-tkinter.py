import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk, scrolledtext

from PIL import Image, ImageTk

from GUITools.functional import detect, train_model
from RailwayDetection.utility import array255topil, array255toheatmap


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RailwayDetection")
        self.geometry("600x700")
        self.resizable(False, False)                                # 禁止调整窗口大小

        self.notebook = ttk.Notebook(self)                          # 页签容器
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.detect_tab = DetectTab(self.notebook, self)
        self.train_tab  = BuildModTab(self.notebook, self)

        self.notebook.add(self.detect_tab, text="Detect")
        self.notebook.add(self.train_tab, text="Build Model")


class DetectTab(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._image_path: str | None = None
        self._build()

    def _build(self):

        main_frame = self
        v1_frame = tk.Frame(main_frame)
        v2_frame = tk.Frame(main_frame)
        v3_frame = tk.Frame(main_frame)

        h1_frame = tk.Frame(v1_frame)
        entry_frame_model, self.entry_model = self.__build_entry(h1_frame, text="Model Path")
        entry_frame_model.pack(side="top", padx=5, pady=5)

        detect_btn = self.__build_button(v1_frame)
        h1_frame.pack(side="left", fill="x", padx=5, pady=5)
        detect_btn.pack(side="left", padx=5, pady=5)

        image_frame_input, self.image_label_input = self.__build_image_panel(
            v2_frame, title="Input-Image", on_click=self.__on_input_image_click)
        image_frame_result, self.image_label_result = self.__build_image_panel(
            v2_frame, title="Detected-Result")
        image_frame_input.pack(side="left", fill='both', expand=True, padx=25, pady=5)
        image_frame_result.pack(side="left", fill='both', expand=True, padx=25, pady=5)

        log_frame, self.log_text = self.__build_log_window(v3_frame)
        log_frame.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        v1_frame.pack(side="top", fill="x", padx=5, pady=5)
        v2_frame.pack(side="top", fill="both", expand=True, padx=5, pady=5)
        v3_frame.pack(side="top", fill="x", padx=5, pady=5)

        self.image_label_input.config(text="Click to select")


    def __build_button(self, master: tk.Frame | None) -> ttk.Button:
        btn = ttk.Button(
            master if master is not None else self,
            text="START-DETECT",
            command=self.__btn_detect_on_click)
        return btn

    def __build_log_window(self, master: tk.Frame | None = None) -> tuple[ttk.LabelFrame, scrolledtext.ScrolledText]:
        # 先创建一个 LabelFrame 作为日志区的外框
        log_frame = ttk.LabelFrame(
            master if master is not None else self,
            text = "Log",
            padding=5)
        logtext = scrolledtext.ScrolledText(
            master = log_frame,
            height = 12,
            wrap = tk.WORD,
            state = 'normal')
        logtext.pack(fill="both", expand=True)
        return log_frame, logtext

    def __build_entry(self, master: tk.Frame | None = None, text: str = "") -> tuple[ttk.Frame, tk.StringVar]:
        # 外层容器 水平排列
        frame = ttk.Frame(master if master is not None else self)
        # 标签
        label = ttk.Label(frame, text=text)
        label.pack(side="left", padx=5)
        # 输入框
        entry_var = tk.StringVar(value="test/model.npz")
        entry = ttk.Entry(frame, textvariable=entry_var, width=40)
        entry.pack(side="left", padx=5, fill="x", expand=True)
        return frame, entry_var

    def __build_image_panel(self, master: tk.Frame | None = None, title: str = "",
                            on_click=None) -> tuple[ttk.LabelFrame, tk.Label]:
        frame = ttk.LabelFrame(
            master if master is not None else self,
            text = title,
            padding=5,
            )
        # 用于显示图片的 Label
        img_label = tk.Label(frame, background="#ddd")
        if on_click is not None: img_label.bind("<Button-1>", on_click)
        img_label.pack(fill="both", expand=True)
        return frame, img_label

    def __on_input_image_click(self, _event=None):
        path = filedialog.askopenfilename(
            title="Select detection image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")])
        if path: self._set_input_image(path)

    def _set_input_image(self, path: str):
        path = path.strip().strip('"')
        if not Path(path).is_file():
            self.__log(f"Image file does not exist: {path}")
            return
        self._image_path = path
        self.__log(f"Selected detection image: {path}")
        try:
            preview = Image.open(path)
            self.__show_image(self.image_label_input, preview)
        except Exception as exc:
            self.__log(f"Failed to preview image: {exc}")

    def __btn_detect_on_click(self):
        if self._image_path is None:
            self.__log("Please select a detection image first.")
            return
        result = detect(self.entry_model.get().strip(), self._image_path, log=self.__log)
        if result is None: return
        result_img, boxes, info = result
        self.__show_image(self.image_label_result, result_img)
        self.__log(f"Detected {len(boxes)} anomaly region(s), "
                  f"max_z = {info['max_z']:.2f}, z_p99 = {info['z_p99']:.2f}")
        for idx, (x, y, w, h) in enumerate(boxes, start=1):
            self.__log(f"  [{idx}] x={x}, y={y}, w={w}, h={h}")

    def __show_image(self, label: tk.Label, image: Image.Image):
        w, h = label.winfo_width(), label.winfo_height()
        if w < 10 or h < 10: w, h = 320, 240
        max_w, max_h = max(w - 12, 120), max(h - 12, 120)
        copy = image.copy()
        copy.thumbnail((max_w, max_h))
        photo = ImageTk.PhotoImage(copy)
        label.config(image=photo)
        label.image = photo                     # type: ignore

    def __log(self, message: str):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)


class BuildModTab(tk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._build()

    def _build(self):

        main_frame = self
        v1_frame = tk.Frame(main_frame)
        v2_frame = tk.Frame(main_frame)
        v3_frame = tk.Frame(main_frame)

        h1_frame = tk.Frame(v1_frame)
        entry_frame_model, self.entry_model_load = self.__build_entry(h1_frame, text="Load Model Path", default="")
        entry_frame_save, self.entry_save = self.__build_entry(h1_frame, text="Save Model Path")
        entry_frame_folder, self.entry_image_folder = self.__build_entry(h1_frame, text="Image Folder Path", default="nor_img")
        entry_frame_model.pack(side="top", padx=5, pady=5)
        entry_frame_save.pack(side="top", padx=5, pady=5)
        entry_frame_folder.pack(side="top", padx=5, pady=5)

        build_btn = self.__build_button(v1_frame)
        h1_frame.pack(side="left", fill="x", padx=5, pady=5)
        build_btn.pack(side="left", padx=5, pady=10)

        image_frame_nor, self.imgage_label_nor = self.__build_image_panel(v2_frame, title="Mod-Nor-Image")
        image_frame_sigma, self.imgage_label_sigma = self.__build_image_panel(v2_frame, title="Mod-Sigma-Image")
        image_frame_nor.pack(side="left", fill='both', expand=True, padx=25, pady=5)
        image_frame_sigma.pack(side="left", fill='both', expand=True, padx=25, pady=5)

        log_frame, self.log_text = self.__build_log_window(v3_frame)
        log_frame.pack(side="left", fill="x", expand=True, padx=5, pady=5)

        v1_frame.pack(side="top", fill="x", padx=5, pady=5)
        v2_frame.pack(side="top", fill="both", expand=True, padx=5, pady=5)
        v3_frame.pack(side="top", fill="x", padx=5, pady=5)

    def __build_button(self, master: tk.Frame | None) -> ttk.Button:
        btn = ttk.Button(
            master if master is not None else self,
            text="BUILD-MODEL",
            command=self.__btn_builemod_on_click)
        return btn
    
    def __build_log_window(self, master: tk.Frame | None = None) -> tuple[ttk.LabelFrame, scrolledtext.ScrolledText]:
        # 先创建一个 LabelFrame 作为日志区的外框
        log_frame = ttk.LabelFrame(
            master if master is not None else self,
            text = "Log",
            padding=5)
        logtext = scrolledtext.ScrolledText(
            master = log_frame,
            height = 12,
            wrap = tk.WORD,
            state = 'normal')
        logtext.pack(fill="both", expand=True)
        return log_frame, logtext
    
    def __build_entry(self, master: tk.Frame | None = None, text: str = "",
                      default: str = "model.npz") -> tuple[ttk.Frame, tk.StringVar]:
        # 外层容器 水平排列
        frame = ttk.Frame(master if master is not None else self)
        # 标签
        label = ttk.Label(frame, text=text)
        label.pack(side="left", padx=5)
        # 输入框
        entry_var = tk.StringVar(value=default)
        entry = ttk.Entry(frame, textvariable=entry_var, width=40)
        entry.pack(side="left", padx=5, fill="x", expand=True)
        return frame, entry_var
    
    def __build_image_panel(self, master: tk.Frame | None = None, title: str = "") -> tuple[ttk.LabelFrame, tk.Label]:
        frame = ttk.LabelFrame(
            master if master is not None else self,
            text = title,
            padding=5,
            )
        # 用于显示图片的 Label
        img_label = tk.Label(frame, background="#ddd")
        img_label.pack(fill="both", expand=True)
        return frame, img_label

    def __btn_builemod_on_click(self):
        model_path = self.entry_model_load.get().strip() or None
        save_path = self.entry_save.get().strip()
        image_folder = self.entry_image_folder.get().strip()
        model = train_model(model_path, save_path, image_folder, log=self.__log)
        if model is None:
            return
        mean = model.get_mean()
        sigma = model.get_sigma()
        if mean is not None:
            self.__show_image(self.imgage_label_nor, array255topil(mean))
        if sigma is not None:
            mask = (model.count > 0) if model.count is not None else None
            self.__show_image(self.imgage_label_sigma, array255toheatmap(sigma, mask=mask))

    def __show_image(self, label: tk.Label, image: Image.Image):
        w, h = label.winfo_width(), label.winfo_height()
        if w < 10 or h < 10: w, h = 320, 240
        max_w, max_h = max(w - 12, 120), max(h - 12, 120)
        copy = image.copy()
        copy.thumbnail((max_w, max_h))
        photo = ImageTk.PhotoImage(copy)
        label.config(image=photo)
        label.image = photo                     # type: ignore

    def __log(self, message: str):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

if __name__ == "__main__":
    app = App()
    app.mainloop()
