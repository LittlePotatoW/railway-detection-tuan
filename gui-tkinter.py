import tkinter as tk
from tkinter import ttk, scrolledtext


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RailwayDetection")
        self.geometry("600x600")
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
        self._build()
    def _build(self):
        ...


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
        entry_frame_load, self.entry_load = self.__build_entry(h1_frame, text="Load Path")
        entry_frame_save, self.entry_save = self.__build_entry(h1_frame, text="Save Path")
        entry_frame_load.pack(side="top", padx=5, pady=5)
        entry_frame_save.pack(side="top", padx=5, pady=5)

        build_btn = self.__build_button(v1_frame)
        h1_frame.pack(side="left", fill="x", padx=5, pady=10)
        build_btn.pack(side="left", padx=5, pady=10)

        image_frame_nor, self.imgage_label_nor = self.__build_image_panel(v2_frame, title="Mod-Nor-Image")
        image_frame_sigma, self.imgage_label_sigma = self.__build_image_panel(v2_frame, title="Mod-Sigma-Image")
        image_frame_nor.pack(side="left", fill='both', expand=True, padx=40, pady=10)
        image_frame_sigma.pack(side="left", fill='both', expand=True, padx=40, pady=10)

        log_frame, self.log_text = self.__build_log_window(v3_frame)
        log_frame.pack(side="left", fill="x", expand=True, padx=5, pady=10)

        v1_frame.pack(side="top", fill="x", padx=5, pady=10)
        v2_frame.pack(side="top", fill="both", expand=True, padx=5, pady=10)
        v3_frame.pack(side="top", fill="x", padx=5, pady=10)


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
    
    def __build_entry(self, master: tk.Frame | None = None, text: str = "") -> tuple[ttk.Frame, tk.StringVar]:
        # 外层容器 水平排列
        frame = ttk.Frame(master if master is not None else self)
        # 标签
        label = ttk.Label(frame, text=text)
        label.pack(side="left", padx=5)
        # 输入框
        entry_var = tk.StringVar(value="model.npz")
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
        pass

if __name__ == "__main__":
    root = tk.Tk()
    root.title("测试")
    root.geometry("600x600")
    # 创建 BuildModTab 实例，app 参数传 None（因为未使用）
    tab = BuildModTab(root, None)
    tab.pack(fill="both", expand=True)
    root.mainloop()