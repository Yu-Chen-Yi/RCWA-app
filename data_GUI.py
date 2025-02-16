import sys
import os
import numpy as np
import scipy.io as sio
import torch 
import datetime
from checkmac import *
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QProgressBar,
    QFileDialog,
)
from PySide6.QtGui import QPixmap, QFont, QIcon
from PySide6.QtCore import Qt
from RCWA import RCWA
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from DataVisualize import DataVisualize

class PlotWindow(QWidget):
    """獨立新視窗用於顯示 Matplotlib 圖"""
    def __init__(self, parent=None, figure=None, ax=None):
        super().__init__(parent)
        self.setWindowTitle("New Plot Window")
        self.setWindowFlag(Qt.Window)  # 設定為獨立視窗
        self.figure = figure
        self.ax = ax
        self.initUI()

    def initUI(self):
        # 創建 Matplotlib 圖表
        self.canvas = FigureCanvas(self.figure)

        # 使用 QVBoxLayout 排版圖表
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.plot_windows = []  # 保存所有子視窗的引用，避免被垃圾回收
        self.data_sheet = None
        self.is_paused = False
        self.is_running = False
        self.input_fields = {}
        self.combo_boxes = {}
        self.initUI()

    def initUI(self):
        self.setWindowIcon(QIcon('your_icon.ico'))  # 圖示檔案放在同一個資料夾內
        
        # ============== 基本參數區 ==============
        # 偵測裝置擁有的device，存成list放到device_combo選項中
        self.device_label = QLabel("Device:")
        self.device_combo = QComboBox()
        # 偵測裝置：加入 CPU 和 GPU 型號名稱
        device_list = ["CPU (Default)"]

        # 加入 GPU 型號名稱
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_list.append(device_name)
        self.device_combo.addItems(device_list)
        # 形狀選擇
        self.shape_type_label = QLabel("Shape Type:")
        self.shape_type_combo = QComboBox()
        self.shape_type_combo.addItems(["rectangle", "ellipse", "circle","rhombus","square","cross","hollow_square","hollow_circle"])  
        self.shape_type_combo.setCurrentIndex(0)  # 預設選擇 circle
        self.shape_type_combo.currentIndexChanged.connect(self.on_shape_type_changed) 

        # harmic order
        self.harmonic_order_input = QLineEdit()
        self.harmonic_order_input.setText("7")
        # Wavelength
        self.wavelength_input = QLineEdit()
        self.wavelength_input.setText("940.")
        # Period
        self.period_input = QLineEdit()
        self.period_input.setText("500.")
        self.wave_group = QGroupBox("Basic Parameters")
        wave_form = QFormLayout()
        wave_form.addRow("Harmonic order:", self.harmonic_order_input)
        wave_form.addRow("Wavelength:", self.wavelength_input)
        wave_form.addRow("Period:", self.period_input)
        self.wave_group.setLayout(wave_form)
        
        
        # ========== 讀取 material 資料夾下的所有 .txt 檔案 ==========
        material_dir = "Materials_data"  # 資料夾名稱
        material_files = []
        if os.path.isdir(material_dir):
            material_files = [
                f for f in os.listdir(material_dir) if f.endswith(".txt")
            ]
        # 若此資料夾不存在，material_files 會是空陣列
        # 您也可以在這裡加一個 else 來顯示警告或加入預設選項

        # ---------- Substrate Material 下拉選單 ----------
        substrate_material_label = QLabel("Substrate Material:")
        self.substrate_material_combo = QComboBox()
        for f in material_files:
            self.substrate_material_combo.addItem(f)
        
        # ---------- Metasurface Material 下拉選單 ----------
        # Metasurface Thickness
        self.thickness_input = QLineEdit()
        self.thickness_input.setText("500.")
        metasurface_material_label = QLabel("Metasurface Material:")
        self.metasurface_material_combo = QComboBox()
        for f in material_files:
            self.metasurface_material_combo.addItem(f)
        self.metasurface_group = QGroupBox("Metasurface Parameters")
        meta_form = QFormLayout()
        meta_form.addRow("Material:", self.metasurface_material_combo)
        meta_form.addRow("Thickness:", self.thickness_input)
        self.metasurface_group.setLayout(meta_form)

        # ---------- Metasurface Material 下拉選單 ----------
        # Filling Thickness
        self.filling_thickness_input = QLineEdit()
        self.filling_thickness_input.setText("500.")
        filling_material_label = QLabel("Filling Material:")
        self.filling_material_combo = QComboBox()
        for f in material_files:
            self.filling_material_combo.addItem(f)
        self.filling_group = QGroupBox("Filling Parameters")
        filling_form = QFormLayout()
        filling_form.addRow("Material:", self.filling_material_combo)
        filling_form.addRow("Thickness:", self.filling_thickness_input)
        self.filling_group.setLayout(filling_form)

        # ---------- Slab Material --------
        self.slab_thickness_input = QLineEdit()
        self.slab_thickness_input.setText("500.")
        self.slab_material_combo = QComboBox()
        for f in material_files:
            self.slab_material_combo.addItem(f)
        self.slab_group = QGroupBox("Buffer Parameters")
        slab_form = QFormLayout()
        slab_form.addRow("Material:", self.slab_material_combo)
        slab_form.addRow("Thickness:", self.slab_thickness_input)
        self.slab_group.setLayout(slab_form)

        # ---------- Output Material 下拉選單 ----------
        output_material_label = QLabel("Output Material:")
        self.output_material_combo = QComboBox()
        for f in material_files:
            self.output_material_combo.addItem(f)
        self.inout_group = QGroupBox("Boundary Parameters")
        inout_form = QFormLayout()
        inout_form.addRow("Input Material:", self.substrate_material_combo)
        inout_form.addRow("Output Material:", self.output_material_combo)
        self.inout_group.setLayout(inout_form)
        
        
        # ============== 形狀參數區 (依 shape_type 動態顯示) ==============
        # 1. rectangle: Wx, Wy, theta
        self.rectangle_group = QGroupBox("Rectangle Parameters")
        rect_form = QFormLayout()
        self.rect_Wx_input = QLineEdit()
        self.rect_Wx_input.setText("300.")
        self.rect_Wy_input = QLineEdit()
        self.rect_Wy_input.setText("300.")
        self.rect_theta_input = QLineEdit()
        self.rect_theta_input.setText("0.")
        rect_form.addRow("Wx:", self.rect_Wx_input)
        rect_form.addRow("Wy:", self.rect_Wy_input)
        rect_form.addRow("theta:", self.rect_theta_input)
        self.rectangle_group.setLayout(rect_form)

        # 2. ellipse: Rx, Ry, theta
        self.ellipse_group = QGroupBox("Ellipse Parameters")
        ell_form = QFormLayout()
        self.ell_Rx_input = QLineEdit()
        self.ell_Rx_input.setText("300.")
        self.ell_Ry_input = QLineEdit()
        self.ell_Ry_input.setText("300.")
        self.ell_theta_input = QLineEdit()
        self.ell_theta_input.setText("0.")
        ell_form.addRow("Rx:", self.ell_Rx_input)
        ell_form.addRow("Ry:", self.ell_Ry_input)
        ell_form.addRow("theta:", self.ell_theta_input)
        self.ellipse_group.setLayout(ell_form)

        # 3. circle: R
        self.circle_group = QGroupBox("Circle Parameters")
        cir_form = QFormLayout()
        self.cir_R_input = QLineEdit()
        self.cir_R_input.setText("300.")
        cir_form.addRow("R:", self.cir_R_input)
        self.circle_group.setLayout(cir_form)

        # 4. rhombus: Wx, Wy, theta
        self.rhombus_group = QGroupBox("Rhombus Parameters")
        rhombus_form = QFormLayout()
        self.rhombus_Wx_input = QLineEdit()
        self.rhombus_Wx_input.setText("300.")
        self.rhombus_Wy_input = QLineEdit()
        self.rhombus_Wy_input.setText("300.")
        self.rhombus_theta_input = QLineEdit()
        self.rhombus_theta_input.setText("0.")
        rhombus_form.addRow("Wx:", self.rhombus_Wx_input)
        rhombus_form.addRow("Wy:", self.rhombus_Wy_input)
        rhombus_form.addRow("theta:", self.rhombus_theta_input)
        self.rhombus_group.setLayout(rhombus_form)
        
        # 5. square: W, theta
        self.square_group = QGroupBox("Square Parameters")
        square_form = QFormLayout()
        self.square_W_input = QLineEdit()
        self.square_W_input.setText("300.")
        self.square_theta_input = QLineEdit()
        self.square_theta_input.setText("0.")
        square_form.addRow("W:", self.square_W_input)
        square_form.addRow("theta:", self.square_theta_input)
        self.square_group.setLayout(square_form)
        
        # 6. cross: Wx, Wy, theta
        self.cross_group = QGroupBox("Cross Parameters")
        cross_form = QFormLayout()
        self.cross_Wx_input = QLineEdit()
        self.cross_Wx_input.setText("300.")
        self.cross_Wy_input = QLineEdit()
        self.cross_Wy_input.setText("300.")
        self.cross_theta_input = QLineEdit()
        self.cross_theta_input.setText("0.")
        cross_form.addRow("Wx:", self.cross_Wx_input)
        cross_form.addRow("Wy:", self.cross_Wy_input)
        cross_form.addRow("theta:", self.cross_theta_input)
        self.cross_group.setLayout(cross_form)

        # 7. hollow_square: W, hollow_W, theta
        self.hollow_square_group = QGroupBox("Hollow Square Parameters")
        hollow_square_form = QFormLayout()
        self.hollow_square_W_input = QLineEdit()
        self.hollow_square_W_input.setText("300.")
        self.hollow_square_hollow_W_input = QLineEdit()
        self.hollow_square_hollow_W_input.setText("100.")
        self.hollow_square_theta_input = QLineEdit()
        self.hollow_square_theta_input.setText("0.")
        hollow_square_form.addRow("W:", self.hollow_square_W_input)
        hollow_square_form.addRow("Hollow W:", self.hollow_square_hollow_W_input)
        hollow_square_form.addRow("theta:", self.hollow_square_theta_input)
        self.hollow_square_group.setLayout(hollow_square_form)

        # 8. hollow_circle: R, hollow_R
        self.hollow_circle_group = QGroupBox("Hollow Circle Parameters")
        hollow_circle_form = QFormLayout()
        self.hollow_circle_R_input = QLineEdit()
        self.hollow_circle_R_input.setText("300.")
        self.hollow_circle_hollow_R_input = QLineEdit()
        self.hollow_circle_hollow_R_input.setText("100.")
        hollow_circle_form.addRow("R:", self.hollow_circle_R_input)
        hollow_circle_form.addRow("Hollow R:", self.hollow_circle_hollow_R_input)
        self.hollow_circle_group.setLayout(hollow_circle_form)

        # 預設先顯示 rectangle，其它隱藏
        self.rectangle_group.setVisible(True)
        self.ellipse_group.setVisible(False)
        self.circle_group.setVisible(False)
        self.rhombus_group.setVisible(False)
        self.square_group.setVisible(False)
        self.cross_group.setVisible(False)
        self.hollow_square_group.setVisible(False)
        self.hollow_circle_group.setVisible(False)

        # ============== 形狀參數指南圖片 Label ==============
        # 用來顯示對應的 shape type 指南圖片
        self.shape_guide_label = QLabel()
        self.shape_guide_label.setAlignment(Qt.AlignCenter)
        # 先初始化為第一種 shape
        self.update_shape_guide_image("rectangle")

        # ============== 運算按鈕與結果 ==============
        self.show_button = QPushButton("Show Structure")
        self.show_button.clicked.connect(self.show_structure)
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_rcwa)

        self.transmission_label = QLabel("Transmission: ")
        self.phase_label = QLabel("Phase: ")

        # 假設這裡有一個按鈕，點下去後要做批次計算：
        self.batch_button = QPushButton("Batch Calculation")
        self.batch_button.clicked.connect(self.toggle_batch_calculation)

        # Wavelength 群組
        wavelength_group = QGroupBox("Wavelength")
        wavelength_layout = QFormLayout()
        self.wavelength_min = QLineEdit()
        self.wavelength_max = QLineEdit()
        self.wavelength_n = QLineEdit()
        wavelength_layout.addRow("min", self.wavelength_min)
        wavelength_layout.addRow("max", self.wavelength_max)
        wavelength_layout.addRow("N", self.wavelength_n)
        wavelength_group.setLayout(wavelength_layout)

        # Period 群組
        period_group = QGroupBox("Period")
        period_layout = QFormLayout()
        self.period_min = QLineEdit()
        self.period_max = QLineEdit()
        self.period_n = QLineEdit()
        period_layout.addRow("min", self.period_min)
        period_layout.addRow("max", self.period_max)
        period_layout.addRow("N", self.period_n)
        period_group.setLayout(period_layout)

        # Metasurface Thickness 群組
        thickness_group = QGroupBox("Thickness")
        thickness_layout = QFormLayout()
        self.thickness_min = QLineEdit()
        self.thickness_max = QLineEdit()
        self.thickness_n = QLineEdit()
        thickness_layout.addRow("min", self.thickness_min)
        thickness_layout.addRow("max", self.thickness_max)
        thickness_layout.addRow("N", self.thickness_n)
        thickness_group.setLayout(thickness_layout)

        # retangle, cross Wx 群組
        self.Wx_group = QGroupBox("Wx")
        Wx_layout = QFormLayout()
        self.Wx_min = QLineEdit()
        self.Wx_max = QLineEdit()
        self.Wx_n = QLineEdit()
        Wx_layout.addRow("min", self.Wx_min)
        Wx_layout.addRow("max", self.Wx_max)
        Wx_layout.addRow("N", self.Wx_n)
        self.Wx_group.setLayout(Wx_layout)

        # retangle, cross Wy 群組
        self.Wy_group = QGroupBox("Wy")
        Wy_layout = QFormLayout()
        self.Wy_min = QLineEdit()
        self.Wy_max = QLineEdit()
        self.Wy_n = QLineEdit()
        Wy_layout.addRow("min", self.Wy_min)
        Wy_layout.addRow("max", self.Wy_max)
        Wy_layout.addRow("N", self.Wy_n)
        self.Wy_group.setLayout(Wy_layout)

        # ellipse Rx 群組
        self.Rx_group = QGroupBox("Rx")
        Rx_layout = QFormLayout()
        self.Rx_min = QLineEdit()
        self.Rx_max = QLineEdit()
        self.Rx_n = QLineEdit()
        Rx_layout.addRow("min", self.Rx_min)
        Rx_layout.addRow("max", self.Rx_max)
        Rx_layout.addRow("N", self.Rx_n)
        self.Rx_group.setLayout(Rx_layout)

        # ellipse Ry 群組
        self.Ry_group = QGroupBox("Ry")
        Ry_layout = QFormLayout()
        self.Ry_min = QLineEdit()
        self.Ry_max = QLineEdit()
        self.Ry_n = QLineEdit()
        Ry_layout.addRow("min", self.Ry_min)
        Ry_layout.addRow("max", self.Ry_max)
        Ry_layout.addRow("N", self.Ry_n)
        self.Ry_group.setLayout(Ry_layout)

        # circle R 群組
        self.R_group = QGroupBox("R")
        R_layout = QFormLayout()
        self.R_min = QLineEdit()
        self.R_max = QLineEdit()
        self.R_n = QLineEdit()
        R_layout.addRow("min", self.R_min)
        R_layout.addRow("max", self.R_max)
        R_layout.addRow("N", self.R_n)
        self.R_group.setLayout(R_layout)

        # hollow square hollow_W 群組
        self.hollow_W_group = QGroupBox("Hollow W")
        hollow_W_layout = QFormLayout()
        self.hollow_W_min = QLineEdit()
        self.hollow_W_max = QLineEdit()
        self.hollow_W_n = QLineEdit()
        hollow_W_layout.addRow("min", self.hollow_W_min)
        hollow_W_layout.addRow("max", self.hollow_W_max)
        hollow_W_layout.addRow("N", self.hollow_W_n)
        self.hollow_W_group.setLayout(hollow_W_layout)

        # hollow circle hollow_R 群組
        self.hollow_R_group = QGroupBox("Hollow R")
        hollow_R_layout = QFormLayout()
        self.hollow_R_min = QLineEdit()
        self.hollow_R_max = QLineEdit()
        self.hollow_R_n = QLineEdit()
        hollow_R_layout.addRow("min", self.hollow_R_min)
        hollow_R_layout.addRow("max", self.hollow_R_max)
        hollow_R_layout.addRow("N", self.hollow_R_n)
        self.hollow_R_group.setLayout(hollow_R_layout)

        # theta 群組
        self.theta = QGroupBox("Theta")
        theta_layout = QFormLayout()
        self.theta_min = QLineEdit()
        self.theta_max = QLineEdit()
        self.theta_n = QLineEdit()
        theta_layout.addRow("min", self.theta_min)
        theta_layout.addRow("max", self.theta_max)
        theta_layout.addRow("N", self.theta_n)
        self.theta.setLayout(theta_layout)

        # 預設先顯示 rectangle，其它隱藏
        self.Wx_group.setVisible(True)
        self.Wy_group.setVisible(True)
        self.theta.setVisible(True)
        self.Rx_group.setVisible(False)
        self.Ry_group.setVisible(False)
        self.R_group.setVisible(False)
        self.hollow_W_group.setVisible(False)
        self.hollow_R_group.setVisible(False)

        # 水平佈局：放置 Wavelength 和 Period
        horizontal_layout1 = QHBoxLayout()
        
        horizontal_layout1.addWidget(wavelength_group)
        horizontal_layout1.addWidget(period_group)
        horizontal_layout1.addWidget(thickness_group)

        # 水平佈局：放置 Wx, Wy, Rx, Ry, R
        horizontal_layout2 = QHBoxLayout()
        horizontal_layout2.addWidget(self.Wx_group)
        horizontal_layout2.addWidget(self.Wy_group)
        horizontal_layout2.addWidget(self.Rx_group)
        horizontal_layout2.addWidget(self.Ry_group)
        horizontal_layout2.addWidget(self.R_group)
        horizontal_layout2.addWidget(self.hollow_W_group)
        horizontal_layout2.addWidget(self.hollow_R_group)
        horizontal_layout2.addWidget(self.theta)
        # ============== 佈局 ==============
        # 水平佈局0
        layout0 = QHBoxLayout()
        layout0.addWidget(self.shape_type_label)
        layout0.addWidget(self.shape_type_combo, stretch=1)
        layout0.addWidget(self.device_label)
        layout0.addWidget(self.device_combo, stretch=1)
        # 水平佈局1
        layout1 = QHBoxLayout()
        layout1.addWidget(self.wave_group)
        layout1.addWidget(self.rectangle_group)
        layout1.addWidget(self.ellipse_group)
        layout1.addWidget(self.circle_group)
        layout1.addWidget(self.rhombus_group)
        layout1.addWidget(self.square_group)
        layout1.addWidget(self.cross_group)
        layout1.addWidget(self.hollow_square_group)
        layout1.addWidget(self.hollow_circle_group)
        layout1.addWidget(self.show_button)
        # 水平佈局2
        layout2 = QHBoxLayout()
        layout2.addWidget(self.inout_group)
        layout2.addWidget(self.slab_group)
        layout2.addWidget(self.metasurface_group)
        layout2.addWidget(self.filling_group)
        # 水平佈局3
        layout3 = QHBoxLayout()
        # 執行按鈕、結果
        layout3.addWidget(self.run_button, stretch=1)
        layout3.addWidget(self.transmission_label, stretch=2)
        layout3.addWidget(self.phase_label, stretch=2)

        # 增加一個進度條
        layout_batch = QHBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        # 增加一個按鈕來選擇檔案並保存
        self.save_button = QPushButton("Save Tensors to File", self)
        self.save_button.clicked.connect(self.save_tensors_to_file)
        
        # 按鈕: 打開 DataVisualizer
        self.btn_open_visualizer = QPushButton("開啟 DataVisualizer")
        self.btn_open_visualizer.clicked.connect(self.openDataVisualizer)

        layout_batch.addWidget(self.batch_button)
        layout_batch.addWidget(self.progress_bar)
        layout_batch.addWidget(self.save_button)
        layout_batch.addWidget(self.btn_open_visualizer)
        

        # Batch calculation layout
        LAYOUT_batch = QVBoxLayout()
        LAYOUT_batch.addLayout(horizontal_layout1)
        LAYOUT_batch.addLayout(horizontal_layout2)
        LAYOUT_batch.addLayout(layout_batch)
        
        
        LAYOUT_SETTING = QVBoxLayout()
        LAYOUT_SETTING.addLayout(layout0)
        LAYOUT_SETTING.addLayout(layout1)
        LAYOUT_SETTING.addWidget(self.shape_guide_label)
        LAYOUT_SETTING.addLayout(layout2)
        LAYOUT_SETTING.addLayout(layout3)
        # 垂直佈局
        layout = QVBoxLayout()
        # 形狀選擇
        layout.addLayout(LAYOUT_SETTING)

        # Batch Calculation
        layout.addLayout(LAYOUT_batch)
        self.setLayout(layout)
        self.setWindowTitle("Metasurface Data GUI")
        self.resize(600, 600)

    def toggle_batch_calculation(self):
        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.batch_button.setText("Pause")
            self.batch_calculation()
        else:
            self.is_paused = not self.is_paused
            self.batch_button.setText("Continue" if self.is_paused else "Pause")        

    def on_shape_type_changed(self):
        """
        根據下拉式選單選擇的 shape_type，顯示/隱藏對應的參數群組，
        並更新顯示對應的指南圖片。
        """
        current_shape = self.shape_type_combo.currentText()
        self.rectangle_group.setVisible(False)
        self.ellipse_group.setVisible(False)
        self.circle_group.setVisible(False)
        self.rhombus_group.setVisible(False)
        self.square_group.setVisible(False)
        self.cross_group.setVisible(False)
        self.hollow_circle_group.setVisible(False)
        self.hollow_square_group.setVisible(False)
        self.Wx_group.setVisible(False)
        self.Wy_group.setVisible(False)
        self.theta.setVisible(False)
        self.Rx_group.setVisible(False)
        self.Ry_group.setVisible(False)
        self.R_group.setVisible(False)
        self.hollow_W_group.setVisible(False)
        self.hollow_R_group.setVisible(False)

        if current_shape == "rectangle":
            self.rectangle_group.setVisible(True)
            self.update_shape_guide_image("rectangle")
            self.Wx_group.setVisible(True)
            self.Wy_group.setVisible(True)
            self.theta.setVisible(True)

        elif current_shape == "ellipse":
            self.ellipse_group.setVisible(True)
            self.update_shape_guide_image("ellipse")
            self.Rx_group.setVisible(True)
            self.Ry_group.setVisible(True)
            self.theta.setVisible(True)

        elif current_shape == "circle":
            self.circle_group.setVisible(True)
            self.update_shape_guide_image("circle")
            self.R_group.setVisible(True)
        
        elif current_shape == "rhombus":
            self.rhombus_group.setVisible(True)
            self.update_shape_guide_image("rhombus")
            self.Wx_group.setVisible(True)
            self.Wy_group.setVisible(True)
            self.theta.setVisible(True)

        elif current_shape == "square":
            self.square_group.setVisible(True)
            self.update_shape_guide_image("square")
            self.Wx_group.setVisible(True)
            self.theta.setVisible(True)

        elif current_shape == "cross":
            self.cross_group.setVisible(True)
            self.update_shape_guide_image("cross")
            self.Wx_group.setVisible(True)
            self.Wy_group.setVisible(True)
            self.theta.setVisible(True)

        elif current_shape == "hollow_square":
            self.hollow_square_group.setVisible(True)
            self.update_shape_guide_image("hollow_square")
            self.Wx_group.setVisible(True)
            self.hollow_W_group.setVisible(True)
            self.theta.setVisible(True)

        elif current_shape == "hollow_circle":
            self.hollow_circle_group.setVisible(True)
            self.update_shape_guide_image("hollow_circle")
            self.R_group.setVisible(True)
            self.hollow_R_group.setVisible(True)

    def update_shape_guide_image(self, shape_type):
        """
        根據 shape_type 載入對應的圖片檔。如果檔案不存在，顯示 "Image not found"。
        假設圖片在與程式同一路徑底下，例如：
          - rectangle_guide.png
          - ellipse_guide.png
          - circle_guide.png
        """
        image_map = {
            "rectangle": "rectangle_guide.jpg",
            "ellipse": "ellipse_guide.jpg",
            "circle": "circle_guide.jpg",
            "rhombus": "rhombus_guide.jpg",
            "square": "square_guide.jpg",
            "cross": "cross_guide.jpg",
            "hollow_square": "hollow_square_guide.jpg",
            "hollow_circle": "hollow_circle_guide.jpg",
        }
        image_file = image_map.get(shape_type, None)

        if not image_file:
            self.shape_guide_label.setText("No image mapped.")
            return

        # 判斷路徑是否存在
        if os.path.exists(image_file):
            pixmap = QPixmap(image_file)
            # 調整圖片大小，可依需求修改
            scaled_pixmap = pixmap.scaled(600, 600, Qt.KeepAspectRatio)
            self.shape_guide_label.setPixmap(scaled_pixmap)
        else:
            # 找不到該圖示檔
            self.shape_guide_label.setText(f"Image not found: {image_file}")

    def get_gui_parameters(self):
        """
        獲取 GUI 上的所有參數設置，並以字典形式返回。
        """
        # 抓取裝置self.device_combo選擇的index
        if self.device_combo.currentIndex() > 0:
            device = torch.device(f"cuda:{self.device_combo.currentIndex() - 1}")
        else:
            device = torch.device("cpu")
        # 通用參數
        shape_type = self.shape_type_combo.currentText()
        harmonic_order = int(self.harmonic_order_input.text()) if self.harmonic_order_input.text() else 7
        wavelength = float(self.wavelength_input.text()) if self.wavelength_input.text() else 0.0
        wavelength_min = float(self.wavelength_min.text()) if self.wavelength_min.text() else 0.0
        wavelength_max = float(self.wavelength_max.text()) if self.wavelength_max.text() else 0.0
        wavelength_n = int(self.wavelength_n.text()) if self.wavelength_n.text() else 0
        period = float(self.period_input.text()) if self.period_input.text() else 0.0
        period_min = float(self.period_min.text()) if self.period_min.text() else 0.0
        period_max = float(self.period_max.text()) if self.period_max.text() else 0.0
        period_n = int(self.period_n.text()) if self.period_n.text() else 0
        metasurface_thickness = float(self.thickness_input.text()) if self.thickness_input.text() else 0.0
        metasurface_thickness_max = float(self.thickness_max.text()) if self.thickness_max.text() else 0.0
        metasurface_thickness_min = float(self.thickness_min.text()) if self.thickness_min.text() else 0.0
        metasurface_thickness_n = int(self.thickness_n.text()) if self.thickness_n.text() else 0
        filling_thickness = float(self.filling_thickness_input.text()) if self.filling_thickness_input.text() else 0.0
        slab_thickness = float(self.slab_thickness_input.text()) if self.slab_thickness_input.text() else 0.0
        # 材料選擇
        metasurface_material = self.metasurface_material_combo.currentText()
        substrate_material = self.substrate_material_combo.currentText()
        filling_material = self.filling_material_combo.currentText()
        output_material = self.output_material_combo.currentText()
        slab_material = self.slab_material_combo.currentText()
        # 形狀參數依 shape_type 判斷
        shape_params = {
            "Wx": None,
            "Wx_max": None,
            "Wx_min": None,
            "Wx_n": None,
            "Wy": None,
            "Wy_max": None,
            "Wy_min": None,
            "Wy_n": None,
            "theta": None,
            "theta_max": None,
            "theta_min": None,
            "theta_n": None,
            "Rx": None,
            "Rx_max": None,
            "Rx_min": None,
            "Rx_n": None,
            "Ry": None,
            "Ry_max": None,
            "Ry_min": None,
            "Ry_n": None,
            "R": None,
            "R_max": None,
            "R_min": None,
            "R_n": None,
            "hollow_W": None,
            "hollow_W_max": None,
            "hollow_W_min": None,
            "hollow_W_n": None,
            "hollow_R": None,
            "hollow_R_max": None,
            "hollow_R_min": None,
            "hollow_R_n": None,
        }

        if shape_type == "rectangle":
            shape_params["Wx"] = float(self.rect_Wx_input.text()) if self.rect_Wx_input.text() else 0.0
            shape_params["Wx_max"] = float(self.Wx_max.text()) if self.Wx_max.text() else 0.0
            shape_params["Wx_min"] = float(self.Wx_min.text()) if self.Wx_min.text() else 0.0
            shape_params["Wx_n"] = int(self.Wx_n.text()) if self.Wx_n.text() else 0.0
            shape_params["Wy"] = float(self.rect_Wy_input.text()) if self.rect_Wy_input.text() else 0.0
            shape_params["Wy_max"] = float(self.Wy_max.text()) if self.Wy_max.text() else 0.0
            shape_params["Wy_min"] = float(self.Wy_min.text()) if self.Wy_min.text() else 0.0
            shape_params["Wy_n"] = int(self.Wy_n.text()) if self.Wy_n.text() else 0.0
            shape_params["theta"] = float(self.rect_theta_input.text()) if self.rect_theta_input.text() else 0.0
            shape_params["theta_max"] = float(self.theta_max.text()) if self.theta_max.text() else 0.0
            shape_params["theta_min"] = float(self.theta_min.text()) if self.theta_min.text() else 0.0
            shape_params["theta_n"] = int(self.theta_n.text()) if self.theta_n.text() else 0.0


        elif shape_type == "ellipse":
            shape_params["Rx"] = float(self.ell_Rx_input.text()) if self.ell_Rx_input.text() else 0.0
            shape_params["Rx_max"] = float(self.Rx_max.text()) if self.Rx_max.text() else 0.0
            shape_params["Rx_min"] = float(self.Rx_min.text()) if self.Rx_min.text() else 0.0
            shape_params["Rx_n"] = int(self.Rx_n.text()) if self.Rx_n.text() else 0.0
            shape_params["Ry"] = float(self.ell_Ry_input.text()) if self.ell_Ry_input.text() else 0.0
            shape_params["Ry_max"] = float(self.Ry_max.text()) if self.Ry_max.text() else 0.0
            shape_params["Ry_min"] = float(self.Ry_min.text()) if self.Ry_min.text() else 0.0
            shape_params["Ry_n"] = int(self.Ry_n.text()) if self.Ry_n.text() else 0.0
            shape_params["theta"] = float(self.ell_theta_input.text()) if self.ell_theta_input.text() else 0.0
            shape_params["theta_max"] = float(self.theta_max.text()) if self.theta_max.text() else 0.0
            shape_params["theta_min"] = float(self.theta_min.text()) if self.theta_min.text() else 0.0
            shape_params["theta_n"] = int(self.theta_n.text()) if self.theta_n.text() else 0.0

        elif shape_type == "circle":
            shape_params["R"] = float(self.cir_R_input.text()) if self.cir_R_input.text() else 0.0
            shape_params["R_max"] = float(self.R_max.text()) if self.R_max.text() else 0.0
            shape_params["R_min"] = float(self.R_min.text()) if self.R_min.text() else 0.0
            shape_params["R_n"] = int(self.R_n.text()) if self.R_n.text() else 0.0

        elif shape_type == "rhombus":
            shape_params["Wx"] = float(self.rhombus_Wx_input.text()) if self.rhombus_Wx_input.text() else 0.0
            shape_params["Wx_max"] = float(self.Wx_max.text()) if self.Wx_max.text() else 0.0
            shape_params["Wx_min"] = float(self.Wx_min.text()) if self.Wx_min.text() else 0.0
            shape_params["Wx_n"] = int(self.Wx_n.text()) if self.Wx_n.text() else 0.0
            shape_params["Wy"] = float(self.rhombus_Wy_input.text()) if self.rhombus_Wy_input.text() else 0.0
            shape_params["Wy_max"] = float(self.Wy_max.text()) if self.Wy_max.text() else 0.0
            shape_params["Wy_min"] = float(self.Wy_min.text()) if self.Wy_min.text() else 0.0
            shape_params["Wy_n"] = int(self.Wy_n.text()) if self.Wy_n.text() else 0.0
            shape_params["theta"] = float(self.rhombus_theta_input.text()) if self.rhombus_theta_input.text() else 0.0
            shape_params["theta_max"] = float(self.theta_max.text()) if self.theta_max.text() else 0.0
            shape_params["theta_min"] = float(self.theta_min.text()) if self.theta_min.text() else 0.0
            shape_params["theta_n"] = int(self.theta_n.text()) if self.theta_n.text() else 0.0
        
        elif shape_type == "square":
            shape_params["Wx"] = float(self.square_W_input.text()) if self.square_W_input.text() else 0.0
            shape_params["Wx_max"] = float(self.Wx_max.text()) if self.Wx_max.text() else 0.0
            shape_params["Wx_min"] = float(self.Wx_min.text()) if self.Wx_min.text() else 0.0
            shape_params["Wx_n"] = int(self.Wx_n.text()) if self.Wx_n.text() else 0.0
            shape_params["theta"] = float(self.square_theta_input.text()) if self.square_theta_input.text() else 0.0
            shape_params["theta_max"] = float(self.theta_max.text()) if self.theta_max.text() else 0.0
            shape_params["theta_min"] = float(self.theta_min.text()) if self.theta_min.text() else 0.0
            shape_params["theta_n"] = int(self.theta_n.text()) if self.theta_n.text() else 0.0
        
        elif shape_type == "cross":
            shape_params["Wx"] = float(self.cross_Wx_input.text()) if self.cross_Wx_input.text() else 0.0
            shape_params["Wx_max"] = float(self.Wx_max.text()) if self.Wx_max.text() else 0.0
            shape_params["Wx_min"] = float(self.Wx_min.text()) if self.Wx_min.text() else 0.0
            shape_params["Wx_n"] = int(self.Wx_n.text()) if self.Wx_n.text() else 0.0
            shape_params["Wy"] = float(self.cross_Wy_input.text()) if self.cross_Wy_input.text() else 0.0
            shape_params["Wy_max"] = float(self.Wy_max.text()) if self.Wy_max.text() else 0.0
            shape_params["Wy_min"] = float(self.Wy_min.text()) if self.Wy_min.text() else 0.0
            shape_params["Wy_n"] = int(self.Wy_n.text()) if self.Wy_n.text() else 0.0
            shape_params["theta"] = float(self.cross_theta_input.text()) if self.cross_theta_input.text() else 0.0
            shape_params["theta_max"] = float(self.theta_max.text()) if self.theta_max.text() else 0.0
            shape_params["theta_min"] = float(self.theta_min.text()) if self.theta_min.text() else 0.0
            shape_params["theta_n"] = int(self.theta_n.text()) if self.theta_n.text() else 0.0

        elif shape_type == "hollow_square":
            shape_params["Wx"] = float(self.hollow_square_W_input.text()) if self.hollow_square_W_input.text() else 0.0
            shape_params["Wx_max"] = float(self.Wx_max.text()) if self.Wx_max.text() else 0.0
            shape_params["Wx_min"] = float(self.Wx_min.text()) if self.Wx_min.text() else 0.0
            shape_params["Wx_n"] = int(self.Wx_n.text()) if self.Wx_n.text() else 0.0
            shape_params["hollow_W"] = float(self.hollow_square_hollow_W_input.text()) if self.hollow_square_hollow_W_input.text() else 0.0
            shape_params["hollow_W_max"] = float(self.hollow_W_max.text()) if self.hollow_W_max.text() else 0.0
            shape_params["hollow_W_min"] = float(self.hollow_W_min.text()) if self.hollow_W_min.text() else 0.0
            shape_params["hollow_W_n"] = int(self.hollow_W_n.text()) if self.hollow_W_n.text() else 0.0
            shape_params["theta"] = float(self.hollow_square_theta_input.text()) if self.hollow_square_theta_input.text() else 0.0
            shape_params["theta_max"] = float(self.theta_max.text()) if self.theta_max.text() else 0.0
            shape_params["theta_min"] = float(self.theta_min.text()) if self.theta_min.text() else 0.0
            shape_params["theta_n"] = int(self.theta_n.text()) if self.theta_n.text() else 0.0
        
        elif shape_type == "hollow_circle":
            shape_params["R"] = float(self.hollow_circle_R_input.text()) if self.hollow_circle_R_input.text() else 0.0
            shape_params["R_max"] = float(self.R_max.text()) if self.R_max.text() else 0.0
            shape_params["R_min"] = float(self.R_min.text()) if self.R_min.text() else 0.0
            shape_params["R_n"] = int(self.R_n.text()) if self.R_n.text() else 0.0  
            shape_params["hollow_R"] = float(self.hollow_circle_hollow_R_input.text()) if self.hollow_circle_hollow_R_input.text() else 0.0
            shape_params["hollow_R_max"] = float(self.hollow_R_max.text()) if self.hollow_R_max.text() else 0.0
            shape_params["hollow_R_min"] = float(self.hollow_R_min.text()) if self.hollow_R_min.text() else 0.0
            shape_params["hollow_R_n"] = int(self.hollow_R_n.text()) if self.hollow_R_n.text() else 0.0
        
        # 返回所有參數作為字典
        return {
            "harmonic_order": harmonic_order, #QLineEdit
            "device" : device,                #QComboBox
            "shape_type": shape_type,         #QComboBox
            "wavelength": wavelength,         #QLineEdit
            "wavelength_min": wavelength_min, #QLineEdit
            "wavelength_max": wavelength_max, #QLineEdit
            "wavelength_n": wavelength_n,     #QLineEdit
            "period": period,                 #QLineEdit
            "period_min": period_min,         #QLineEdit
            "period_max": period_max,         #QLineEdit
            "period_n": period_n,             #QLineEdit
            "metasurface_thickness": metasurface_thickness,#QLineEdit
            "metasurface_thickness_max": metasurface_thickness_max,#QLineEdit
            "metasurface_thickness_min": metasurface_thickness_min,#QLineEdit
            "metasurface_thickness_n": metasurface_thickness_n,#QLineEdit
            "filling_thickness": filling_thickness,#QLineEdit
            "slab_thickness": slab_thickness,
            "metasurface_material": metasurface_material,#QComboBox
            "substrate_material": substrate_material,#QComboBox
            "slab_material": slab_material,
            "filling_material": filling_material,#QComboBox
            "output_material": output_material,#QComboBox
            **shape_params
        }

    def show_structure(self):
        """
        從介面讀取參數，呼叫 RCWA 類別進行計算，並顯示結構。
        """
        # 獲取 GUI 參數
        params = self.get_gui_parameters()

        # 建立 RCWA 物件並顯示結構
        rcwa_obj = RCWA(
            device=params["device"],
            shape_type=params["shape_type"],
            harmonic_order=params["harmonic_order"],
            wavelength=params["wavelength"],
            period=params["period"],
            substrate_material=params["substrate_material"],
            slab_material=params["slab_material"],
            slab_thickness=params["slab_thickness"],
            metasurface_material=params["metasurface_material"],
            metasurface_thickness=params["metasurface_thickness"],
            filling_material=params["filling_material"],
            filling_thickness=params["filling_thickness"],
            output_material=params["output_material"],
            Wx=params["Wx"],
            Wy=params["Wy"],
            theta=params["theta"],
            Rx=params["Rx"],
            Ry=params["Ry"],
            R=params["R"],
            hollow_W=params["hollow_W"],
            hollow_R=params["hollow_R"]
        )
        figure, ax = rcwa_obj.show_structure()
        new_window = PlotWindow(figure=figure, ax=ax)  # 創建新的 PlotWindow 視窗
        self.plot_windows.append(new_window)  # 保存引用，防止被垃圾回收
        new_window.show()  # 顯示視窗

    def run_rcwa(self):
        """
        從介面讀取參數，呼叫 RCWA 類別進行計算，並顯示結果。
        """
        # 獲取 GUI 參數
        params = self.get_gui_parameters()

        # 建立 RCWA 物件並進行計算
        rcwa_obj = RCWA(
            device=params["device"],
            shape_type=params["shape_type"],
            harmonic_order=params["harmonic_order"],
            wavelength=params["wavelength"],
            period=params["period"],
            substrate_material=params["substrate_material"],
            slab_material=params["slab_material"],
            slab_thickness=params["slab_thickness"],
            metasurface_material=params["metasurface_material"],
            metasurface_thickness=params["metasurface_thickness"],
            filling_material=params["filling_material"],
            filling_thickness=params["filling_thickness"],
            output_material=params["output_material"],
            Wx=params["Wx"],
            Wy=params["Wy"],
            theta=params["theta"],
            Rx=params["Rx"],
            Ry=params["Ry"],
            R=params["R"],
            hollow_W=params["hollow_W"],
            hollow_R=params["hollow_R"]
        )
        txx, txy, tyx, tyy = rcwa_obj.get_Sparameter()
        transmission_x = torch.abs(txx)**2
        transmission_y = torch.abs(tyy)**2
        phase_x = torch.angle(txx)
        phase_y = torch.angle(tyy)
        tRL = (txx - tyy) - 1j * (txy + tyx)
        tRR = (txx + tyy) + 1j * (txy - tyx)
        tLR = (txx - tyy) + 1j * (txy + tyx)
        tLL = (txx + tyy) - 1j * (txy - tyx)
        L_PCE = torch.abs(tRL)**2
        R_PCE = torch.abs(tLR)**2
        phase_R = torch.angle(tRL)
        phase_L = torch.angle(tLR)
        # 更新顯示結果
        self.transmission_label.setText(f"Transmission: {transmission_x}")
        self.phase_label.setText(f"Phase: {phase_x} radians")

    def batch_calculation(self):
        """
        1. 從使用者介面取得 基本參數 (shape_type, wavelength, period, thickness, material... )。
        2. 取得 Wx, Wy, theta 的掃描範圍 (下界、上界、步進)。
        3. 迴圈掃描所有組合，呼叫 RCWA 做計算，並將結果儲存或顯示。
        """
        """
        從介面讀取參數，呼叫 RCWA 類別進行計算，並顯示結果。
        """
        # 獲取 GUI 參數
        params = self.get_gui_parameters()
        # wavelength list
        wavelength_list = np.linspace(params["wavelength_min"], params["wavelength_max"], params["wavelength_n"])
        # period list
        period_list = np.linspace(params["period_min"], params["period_max"], params["period_n"])
        # thickness list
        thickness_list = np.linspace(params["metasurface_thickness_min"], params["metasurface_thickness_max"], params["metasurface_thickness_n"])
        # 初始化進度條
        current_iteration = 0

        if params["shape_type"] == "rectangle" or params["shape_type"] == "rhombus"  or params["shape_type"] == "cross":
            # Wx list
            Wx_list = np.linspace(params["Wx_min"], params["Wx_max"], params["Wx_n"])
            # Wy list
            Wy_list = np.linspace(params["Wy_min"], params["Wy_max"], params["Wy_n"])
            # theta list
            theta_list = np.linspace(params["theta_min"], params["theta_max"], params["theta_n"])
            # 計算總迴圈數
            total_iterations = (
                len(wavelength_list) * len(period_list) * len(thickness_list) *
                len(Wx_list) * len(Wy_list) * len(theta_list)
            )
            dimension_names = [
                "Wavelength (nm)", "Period (nm)", "Thickness (nm)",
                "Wx (nm)", "Wy (nm)", "Rotation Angle (deg)"
            ]
            # 依序迴圈，計算每一組 RCWA(Wavelength, Period, Thickness, Wx, Wy, theta)存入空矩陣shape為(wavelength_n, period_n, thickness_n, Wx_n, Wy_n, theta_n)
            # 創建一個 tensor，初始化為零
            transmission_tensor = torch.zeros(len(wavelength_list), len(period_list), len(thickness_list), len(Wx_list), len(Wy_list), len(theta_list), 8)
            phase_tensor = torch.zeros_like(transmission_tensor)
            # 進行批次計算
            for i, wavelength in enumerate(wavelength_list):
                for j, period in enumerate(period_list):
                    for k, thickness in enumerate(thickness_list):
                        for l, Wx in enumerate(Wx_list):
                            for m, Wy in enumerate(Wy_list):
                                for n, theta in enumerate(theta_list):
                                    while self.is_paused:
                                        QApplication.processEvents()
                                    if not self.is_running:
                                        return
                                    # 模擬 RCWA 計算
                                    rcwa_obj = RCWA(
                                        device=params["device"],
                                        shape_type=params["shape_type"],
                                        harmonic_order=params["harmonic_order"],
                                        wavelength=wavelength,
                                        period=period,
                                        substrate_material=params["substrate_material"],
                                        slab_material=params["slab_material"],
                                        slab_thickness=params["slab_thickness"],
                                        metasurface_material=params["metasurface_material"],
                                        metasurface_thickness=thickness,
                                        filling_material=params["filling_material"],
                                        filling_thickness=params["filling_thickness"],
                                        output_material=params["output_material"],
                                        Wx=Wx,
                                        Wy=Wy,
                                        theta=theta,
                                    )
                                    txx, txy, tyx, tyy = rcwa_obj.get_Sparameter()
                                    transmission_xx = torch.abs(txx)**2
                                    transmission_yy = torch.abs(tyy)**2
                                    phase_xx = torch.angle(txx)
                                    phase_yy = torch.angle(tyy)
                                    transmission_xy = torch.abs(txy)**2
                                    transmission_yx = torch.abs(tyx)**2
                                    phase_xy = torch.angle(txy)
                                    phase_yx = torch.angle(tyx)
                                    tRL = 0.5*((txx - tyy) - 1j * (txy + tyx))
                                    tRR = 0.5*((txx + tyy) + 1j * (txy - tyx))
                                    tLR = 0.5*((txx - tyy) + 1j * (txy + tyx))
                                    tLL = 0.5*((txx + tyy) - 1j * (txy - tyx))
                                    transmission_RL = torch.abs(tRL)**2
                                    transmission_LR = torch.abs(tLR)**2
                                    phase_RL = torch.angle(tRL)
                                    phase_LR = torch.angle(tLR)
                                    transmission_RR = torch.abs(tRR)**2
                                    transmission_LL = torch.abs(tLL)**2
                                    phase_RR = torch.angle(tRR)
                                    phase_LL = torch.angle(tLL)

                                    transmission_tensor[i, j, k, l, m, n, 0] = transmission_xx
                                    phase_tensor[i, j, k, l, m, n, 0] = phase_xx
                                    transmission_tensor[i, j, k, l, m, n, 1] = transmission_yx
                                    phase_tensor[i, j, k, l, m, n, 1] = phase_yx
                                    transmission_tensor[i, j, k, l, m, n, 2] = transmission_xy
                                    phase_tensor[i, j, k, l, m, n, 2] = phase_xy
                                    transmission_tensor[i, j, k, l, m, n, 3] = transmission_yy
                                    phase_tensor[i, j, k, l, m, n, 3] = phase_yy
                                    transmission_tensor[i, j, k, l, m, n, 4] = transmission_LL
                                    phase_tensor[i, j, k, l, m, n, 4] = phase_LL
                                    transmission_tensor[i, j, k, l, m, n, 5] = transmission_RL
                                    phase_tensor[i, j, k, l, m, n, 5] = phase_RL
                                    transmission_tensor[i, j, k, l, m, n, 6] = transmission_LR
                                    phase_tensor[i, j, k, l, m, n, 6] = phase_LR
                                    transmission_tensor[i, j, k, l, m, n, 7] = transmission_RR
                                    phase_tensor[i, j, k, l, m, n, 7] = phase_RR

                                    # 更新進度條
                                    current_iteration += 1
                                    progress = int((current_iteration / total_iterations) * 100)
                                    self.progress_bar.setValue(progress)

                                    # 允許 UI 更新
                                    QApplication.processEvents()
            self.data_sheet = {
                "shape_type": params["shape_type"],
                "Dimension_name": dimension_names,
                "Wavelength": wavelength_list,
                "Period": period_list,
                "Thickness": thickness_list,
                "Wx": Wx_list,
                "Wy": Wy_list,
                "Theta": theta_list,
                "transmission_tensor": transmission_tensor.cpu().numpy(),
                "phase_tensor": phase_tensor.cpu().numpy()
            }
        elif params["shape_type"] == "ellipse":
            # Rx list
            Rx_list = np.linspace(params["Rx_min"], params["Rx_max"], params["Rx_n"])
            # Ry list
            Ry_list = np.linspace(params["Ry_min"], params["Ry_max"], params["Ry_n"])
            # theta list
            theta_list = np.linspace(params["theta_min"], params["theta_max"], params["theta_n"])
            # 計算總迴圈數
            total_iterations = (
                len(wavelength_list) * len(period_list) * len(thickness_list) *
                len(Rx_list) * len(Ry_list) * len(theta_list)
            )
            dimension_names = [
                "Wavelength (nm)", "Period (nm)", "Thickness (nm)",
                "Rx (nm)", "Ry (nm)", "Rotation Angle (deg)"
            ]
            # 依序迴圈，計算每一組 RCWA(Wavelength, Period, Thickness, Wx, Wy, theta)存入空矩陣shape為(wavelength_n, period_n, thickness_n, Wx_n, Wy_n, theta_n)
            # 創建一個 tensor，初始化為零
            transmission_tensor = torch.zeros(len(wavelength_list), len(period_list), len(thickness_list), len(Rx_list), len(Ry_list), len(theta_list), 8)
            phase_tensor = torch.zeros_like(transmission_tensor)
            # 進行批次計算
            for i, wavelength in enumerate(wavelength_list):
                for j, period in enumerate(period_list):
                    for k, thickness in enumerate(thickness_list):
                        for l, Rx in enumerate(Rx_list):
                            for m, Ry in enumerate(Ry_list):
                                for n, theta in enumerate(theta_list):
                                    while self.is_paused:
                                        QApplication.processEvents()
                                    if not self.is_running:
                                        return
                                    # 模擬 RCWA 計算
                                    rcwa_obj = RCWA(
                                        device=params["device"],
                                        shape_type=params["shape_type"],
                                        harmonic_order=params["harmonic_order"],
                                        wavelength=wavelength,
                                        period=period,
                                        substrate_material=params["substrate_material"],
                                        slab_material=params["slab_material"],
                                        slab_thickness=params["slab_thickness"],
                                        metasurface_material=params["metasurface_material"],
                                        metasurface_thickness=thickness,
                                        filling_material=params["filling_material"],
                                        filling_thickness=params["filling_thickness"],
                                        output_material=params["output_material"],
                                        Rx=Rx,
                                        Ry=Ry,
                                        theta=theta,
                                    )
                                    txx, txy, tyx, tyy = rcwa_obj.get_Sparameter()
                                    transmission_xx = torch.abs(txx)**2
                                    transmission_yy = torch.abs(tyy)**2
                                    phase_xx = torch.angle(txx)
                                    phase_yy = torch.angle(tyy)
                                    transmission_xy = torch.abs(txy)**2
                                    transmission_yx = torch.abs(tyx)**2
                                    phase_xy = torch.angle(txy)
                                    phase_yx = torch.angle(tyx)
                                    tRL = 0.5*((txx - tyy) - 1j * (txy + tyx))
                                    tRR = 0.5*((txx + tyy) + 1j * (txy - tyx))
                                    tLR = 0.5*((txx - tyy) + 1j * (txy + tyx))
                                    tLL = 0.5*((txx + tyy) - 1j * (txy - tyx))
                                    transmission_RL = torch.abs(tRL)**2
                                    transmission_LR = torch.abs(tLR)**2
                                    phase_RL = torch.angle(tRL)
                                    phase_LR = torch.angle(tLR)
                                    transmission_RR = torch.abs(tRR)**2
                                    transmission_LL = torch.abs(tLL)**2
                                    phase_RR = torch.angle(tRR)
                                    phase_LL = torch.angle(tLL)

                                    transmission_tensor[i, j, k, l, m, n, 0] = transmission_xx
                                    phase_tensor[i, j, k, l, m, n, 0] = phase_xx
                                    transmission_tensor[i, j, k, l, m, n, 1] = transmission_yx
                                    phase_tensor[i, j, k, l, m, n, 1] = phase_yx
                                    transmission_tensor[i, j, k, l, m, n, 2] = transmission_xy
                                    phase_tensor[i, j, k, l, m, n, 2] = phase_xy
                                    transmission_tensor[i, j, k, l, m, n, 3] = transmission_yy
                                    phase_tensor[i, j, k, l, m, n, 3] = phase_yy
                                    transmission_tensor[i, j, k, l, m, n, 4] = transmission_LL
                                    phase_tensor[i, j, k, l, m, n, 4] = phase_LL
                                    transmission_tensor[i, j, k, l, m, n, 5] = transmission_RL
                                    phase_tensor[i, j, k, l, m, n, 5] = phase_RL
                                    transmission_tensor[i, j, k, l, m, n, 6] = transmission_LR
                                    phase_tensor[i, j, k, l, m, n, 6] = phase_LR
                                    transmission_tensor[i, j, k, l, m, n, 7] = transmission_RR
                                    phase_tensor[i, j, k, l, m, n, 7] = phase_RR

                                    # 更新進度條
                                    current_iteration += 1
                                    progress = int((current_iteration / total_iterations) * 100)
                                    self.progress_bar.setValue(progress)

                            # 允許 UI 更新
                            QApplication.processEvents()
            self.data_sheet = {
                "shape_type": params["shape_type"],
                "Dimension_name": dimension_names,
                "Wavelength": wavelength_list,
                "Period": period_list,
                "Thickness": thickness_list,
                "Rx": Rx_list,
                "Ry": Ry_list,
                "Theta": theta_list,
                "transmission_tensor": transmission_tensor.cpu().numpy(),
                "phase_tensor": phase_tensor.cpu().numpy()
            }
        elif params["shape_type"] == "circle":
            # R list
            R_list = np.linspace(params["R_min"], params["R_max"], params["R_n"])
            # 計算總迴圈數
            total_iterations = (
                len(wavelength_list) * len(period_list) * len(thickness_list) *
                len(R_list)
            )
            dimension_names = [
                "Wavelength (nm)", "Period (nm)", "Thickness (nm)",
                "R (nm)"
            ]
            # 依序迴圈，計算每一組 RCWA(Wavelength, Period, Thickness, Wx, Wy, theta)存入空矩陣shape為(wavelength_n, period_n, thickness_n, Wx_n, Wy_n, theta_n)
            # 創建一個 tensor，初始化為零
            transmission_tensor = torch.zeros(len(wavelength_list), len(period_list), len(thickness_list), len(R_list), 8)
            phase_tensor = torch.zeros_like(transmission_tensor)
            # 進行批次計算
            for i, wavelength in enumerate(wavelength_list):
                for j, period in enumerate(period_list):
                    for k, thickness in enumerate(thickness_list):
                        for l, R in enumerate(R_list):
                                    while self.is_paused:
                                        QApplication.processEvents()
                                    if not self.is_running:
                                        return
                                    # 模擬 RCWA 計算
                                    rcwa_obj = RCWA(
                                        device=params["device"],
                                        shape_type=params["shape_type"],
                                        harmonic_order=params["harmonic_order"],
                                        wavelength=wavelength,
                                        period=period,
                                        substrate_material=params["substrate_material"],
                                        slab_material=params["slab_material"],
                                        slab_thickness=params["slab_thickness"],
                                        metasurface_material=params["metasurface_material"],
                                        metasurface_thickness=thickness,
                                        filling_material=params["filling_material"],
                                        filling_thickness=params["filling_thickness"],
                                        output_material=params["output_material"],
                                        R=R,
                                    )
                                    txx, txy, tyx, tyy = rcwa_obj.get_Sparameter()
                                    transmission_xx = torch.abs(txx)**2
                                    transmission_yy = torch.abs(tyy)**2
                                    phase_xx = torch.angle(txx)
                                    phase_yy = torch.angle(tyy)
                                    transmission_xy = torch.abs(txy)**2
                                    transmission_yx = torch.abs(tyx)**2
                                    phase_xy = torch.angle(txy)
                                    phase_yx = torch.angle(tyx)
                                    tRL = 0.5*((txx - tyy) - 1j * (txy + tyx))
                                    tRR = 0.5*((txx + tyy) + 1j * (txy - tyx))
                                    tLR = 0.5*((txx - tyy) + 1j * (txy + tyx))
                                    tLL = 0.5*((txx + tyy) - 1j * (txy - tyx))
                                    transmission_RL = torch.abs(tRL)**2
                                    transmission_LR = torch.abs(tLR)**2
                                    phase_RL = torch.angle(tRL)
                                    phase_LR = torch.angle(tLR)
                                    transmission_RR = torch.abs(tRR)**2
                                    transmission_LL = torch.abs(tLL)**2
                                    phase_RR = torch.angle(tRR)
                                    phase_LL = torch.angle(tLL)

                                    transmission_tensor[i, j, k, l, 0] = transmission_xx
                                    phase_tensor[i, j, k, l, 0] = phase_xx
                                    transmission_tensor[i, j, k, l, 1] = transmission_yx
                                    phase_tensor[i, j, k, l, 1] = phase_yx
                                    transmission_tensor[i, j, k, l, 2] = transmission_xy
                                    phase_tensor[i, j, k, l, 2] = phase_xy
                                    transmission_tensor[i, j, k, l, 3] = transmission_yy
                                    phase_tensor[i, j, k, l, 3] = phase_yy
                                    transmission_tensor[i, j, k, l, 4] = transmission_LL
                                    phase_tensor[i, j, k, l, 4] = phase_LL
                                    transmission_tensor[i, j, k, l, 5] = transmission_RL
                                    phase_tensor[i, j, k, l, 5] = phase_RL
                                    transmission_tensor[i, j, k, l, 6] = transmission_LR
                                    phase_tensor[i, j, k, l, 6] = phase_LR
                                    transmission_tensor[i, j, k, l, 7] = transmission_RR
                                    phase_tensor[i, j, k, l, 7] = phase_RR

                                    # 更新進度條
                                    current_iteration += 1
                                    progress = int((current_iteration / total_iterations) * 100)
                                    self.progress_bar.setValue(progress)

                                    # 允許 UI 更新
                                    QApplication.processEvents()
            self.data_sheet = {
                "shape_type": params["shape_type"],
                "Dimension_name": dimension_names,
                "Wavelength": wavelength_list,
                "Period": period_list,
                "Thickness": thickness_list,
                "R": R_list,
                "transmission_tensor": transmission_tensor.cpu().numpy(),
                "phase_tensor": phase_tensor.cpu().numpy()
            }
        elif params["shape_type"] == "square":
            # Wx list
            Wx_list = np.linspace(params["Wx_min"], params["Wx_max"], params["Wx_n"])
            # theta list
            theta_list = np.linspace(params["theta_min"], params["theta_max"], params["theta_n"])
            # 計算總迴圈數
            total_iterations = (
                len(wavelength_list) * len(period_list) * len(thickness_list) *
                len(Wx_list) * len(theta_list)
            )
            dimension_names = [
                "Wavelength (nm)", "Period (nm)", "Thickness (nm)",
                "W (nm)", "Rotation Angle (deg)"
            ]
            # 依序迴圈，計算每一組 RCWA(Wavelength, Period, Thickness, Wx, Wy, theta)存入空矩陣shape為(wavelength_n, period_n, thickness_n, Wx_n, Wy_n, theta_n)
            # 創建一個 tensor，初始化為零
            transmission_tensor = torch.zeros(len(wavelength_list), len(period_list), len(thickness_list), len(Wx_list), len(theta_list), 8)
            phase_tensor = torch.zeros_like(transmission_tensor)
            # 進行批次計算
            for i, wavelength in enumerate(wavelength_list):
                for j, period in enumerate(period_list):
                    for k, thickness in enumerate(thickness_list):
                        for l, Wx in enumerate(Wx_list):
                                for m, theta in enumerate(theta_list):
                                    while self.is_paused:
                                        QApplication.processEvents()
                                    if not self.is_running:
                                        return
                                    # 模擬 RCWA 計算
                                    rcwa_obj = RCWA(
                                        device=params["device"],
                                        shape_type=params["shape_type"],
                                        harmonic_order=params["harmonic_order"],
                                        wavelength=wavelength,
                                        period=period,
                                        substrate_material=params["substrate_material"],
                                        slab_material=params["slab_material"],
                                        slab_thickness=params["slab_thickness"],
                                        metasurface_material=params["metasurface_material"],
                                        metasurface_thickness=thickness,
                                        filling_material=params["filling_material"],
                                        filling_thickness=params["filling_thickness"],
                                        output_material=params["output_material"],
                                        Wx=Wx,
                                        theta=theta,
                                    )
                                    txx, txy, tyx, tyy = rcwa_obj.get_Sparameter()
                                    transmission_xx = torch.abs(txx)**2
                                    transmission_yy = torch.abs(tyy)**2
                                    phase_xx = torch.angle(txx)
                                    phase_yy = torch.angle(tyy)
                                    transmission_xy = torch.abs(txy)**2
                                    transmission_yx = torch.abs(tyx)**2
                                    phase_xy = torch.angle(txy)
                                    phase_yx = torch.angle(tyx)
                                    tRL = 0.5*((txx - tyy) - 1j * (txy + tyx))
                                    tRR = 0.5*((txx + tyy) + 1j * (txy - tyx))
                                    tLR = 0.5*((txx - tyy) + 1j * (txy + tyx))
                                    tLL = 0.5*((txx + tyy) - 1j * (txy - tyx))
                                    transmission_RL = torch.abs(tRL)**2
                                    transmission_LR = torch.abs(tLR)**2
                                    phase_RL = torch.angle(tRL)
                                    phase_LR = torch.angle(tLR)
                                    transmission_RR = torch.abs(tRR)**2
                                    transmission_LL = torch.abs(tLL)**2
                                    phase_RR = torch.angle(tRR)
                                    phase_LL = torch.angle(tLL)

                                    transmission_tensor[i, j, k, l, m, 0] = transmission_xx
                                    phase_tensor[i, j, k, l, m, 0] = phase_xx
                                    transmission_tensor[i, j, k, l, m, 1] = transmission_yx
                                    phase_tensor[i, j, k, l, m, 1] = phase_yx
                                    transmission_tensor[i, j, k, l, m, 2] = transmission_xy
                                    phase_tensor[i, j, k, l, m, 2] = phase_xy
                                    transmission_tensor[i, j, k, l, m, 3] = transmission_yy
                                    phase_tensor[i, j, k, l, m, 3] = phase_yy
                                    transmission_tensor[i, j, k, l, m, 4] = transmission_LL
                                    phase_tensor[i, j, k, l, m, 4] = phase_LL
                                    transmission_tensor[i, j, k, l, m, 5] = transmission_RL
                                    phase_tensor[i, j, k, l, m, 5] = phase_RL
                                    transmission_tensor[i, j, k, l, m, 6] = transmission_LR
                                    phase_tensor[i, j, k, l, m, 6] = phase_LR
                                    transmission_tensor[i, j, k, l, m, 7] = transmission_RR
                                    phase_tensor[i, j, k, l, m, 7] = phase_RR

                                    # 更新進度條
                                    current_iteration += 1
                                    progress = int((current_iteration / total_iterations) * 100)
                                    self.progress_bar.setValue(progress)

                                    # 允許 UI 更新
                                    QApplication.processEvents()
            self.data_sheet = {
                "shape_type": params["shape_type"],
                "Dimension_name": dimension_names,
                "Wavelength": wavelength_list,
                "Period": period_list,
                "Thickness": thickness_list,
                "Wx": Wx_list,
                "Theta": theta_list,
                "transmission_tensor": transmission_tensor.cpu().numpy(),
                "phase_tensor": phase_tensor.cpu().numpy()
            }
        elif params["shape_type"] == "hollow_square":
            # Wx list
            Wx_list = np.linspace(params["Wx_min"], params["Wx_max"], params["Wx_n"])
            # hollow_W list
            hollow_W_list = np.linspace(params["hollow_W_min"], params["hollow_W_max"], params["hollow_W_n"])
            # theta list
            theta_list = np.linspace(params["theta_min"], params["theta_max"], params["theta_n"])
            # 計算總迴圈數
            total_iterations = (
                len(wavelength_list) * len(period_list) * len(thickness_list) *
                len(Wx_list) * len(hollow_W_list) * len(theta_list)
            )
            dimension_names = [
                "Wavelength (nm)", "Period (nm)", "Thickness (nm)",
                "W (nm)", "Hollow Width (nm)", "Rotation Angle (deg)"
            ]
            # 依序迴圈，計算每一組 RCWA(Wavelength, Period, Thickness, Wx, Wy, theta)存入空矩陣shape為(wavelength_n, period_n, thickness_n, Wx_n, Wy_n, theta_n)
            # 創建一個 tensor，初始化為零
            transmission_tensor = torch.zeros(len(wavelength_list), len(period_list), len(thickness_list), len(Wx_list), len(hollow_W_list), len(theta_list), 8)
            phase_tensor = torch.zeros_like(transmission_tensor)
            # 進行批次計算
            for i, wavelength in enumerate(wavelength_list):
                for j, period in enumerate(period_list):
                    for k, thickness in enumerate(thickness_list):
                        for l, Wx in enumerate(Wx_list):
                            for m, hollow_W in enumerate(hollow_W_list):
                                for n, theta in enumerate(theta_list):
                                    while self.is_paused:
                                        QApplication.processEvents()
                                    if not self.is_running:
                                        return
                                    # 模擬 RCWA 計算
                                    rcwa_obj = RCWA(
                                        device=params["device"],
                                        shape_type=params["shape_type"],
                                        harmonic_order=params["harmonic_order"],
                                        wavelength=wavelength,
                                        period=period,
                                        substrate_material=params["substrate_material"],
                                        slab_material=params["slab_material"],
                                        slab_thickness=params["slab_thickness"],
                                        metasurface_material=params["metasurface_material"],
                                        metasurface_thickness=thickness,
                                        filling_material=params["filling_material"],
                                        filling_thickness=params["filling_thickness"],
                                        output_material=params["output_material"],
                                        Wx=Wx,
                                        hollow_W=hollow_W,
                                        theta=theta,
                                    )
                                    txx, txy, tyx, tyy = rcwa_obj.get_Sparameter()
                                    transmission_xx = torch.abs(txx)**2
                                    transmission_yy = torch.abs(tyy)**2
                                    phase_xx = torch.angle(txx)
                                    phase_yy = torch.angle(tyy)
                                    transmission_xy = torch.abs(txy)**2
                                    transmission_yx = torch.abs(tyx)**2
                                    phase_xy = torch.angle(txy)
                                    phase_yx = torch.angle(tyx)
                                    tRL = 0.5*((txx - tyy) - 1j * (txy + tyx))
                                    tRR = 0.5*((txx + tyy) + 1j * (txy - tyx))
                                    tLR = 0.5*((txx - tyy) + 1j * (txy + tyx))
                                    tLL = 0.5*((txx + tyy) - 1j * (txy - tyx))
                                    transmission_RL = torch.abs(tRL)**2
                                    transmission_LR = torch.abs(tLR)**2
                                    phase_RL = torch.angle(tRL)
                                    phase_LR = torch.angle(tLR)
                                    transmission_RR = torch.abs(tRR)**2
                                    transmission_LL = torch.abs(tLL)**2
                                    phase_RR = torch.angle(tRR)
                                    phase_LL = torch.angle(tLL)
                                    
                                    transmission_tensor[i, j, k, l, m, n, 0] = transmission_xx
                                    phase_tensor[i, j, k, l, m, n, 0] = phase_xx
                                    transmission_tensor[i, j, k, l, m, n, 1] = transmission_yx
                                    phase_tensor[i, j, k, l, m, n, 1] = phase_yx
                                    transmission_tensor[i, j, k, l, m, n, 2] = transmission_xy
                                    phase_tensor[i, j, k, l, m, n, 2] = phase_xy
                                    transmission_tensor[i, j, k, l, m, n, 3] = transmission_yy
                                    phase_tensor[i, j, k, l, m, n, 3] = phase_yy
                                    transmission_tensor[i, j, k, l, m, n, 4] = transmission_LL
                                    phase_tensor[i, j, k, l, m, n, 4] = phase_LL
                                    transmission_tensor[i, j, k, l, m, n, 5] = transmission_RL
                                    phase_tensor[i, j, k, l, m, n, 5] = phase_RL
                                    transmission_tensor[i, j, k, l, m, n, 6] = transmission_LR
                                    phase_tensor[i, j, k, l, m, n, 6] = phase_LR
                                    transmission_tensor[i, j, k, l, m, n, 7] = transmission_RR
                                    phase_tensor[i, j, k, l, m, n, 7] = phase_RR

                                    # 更新進度條
                                    current_iteration += 1
                                    progress = int((current_iteration / total_iterations) * 100)
                                    self.progress_bar.setValue(progress)
                        
                                    # 允許 UI 更新
                                    QApplication.processEvents()
            self.data_sheet = {
                "shape_type": params["shape_type"],
                "Dimension_name": dimension_names,
                "Wavelength": wavelength_list,
                "Period": period_list,
                "Thickness": thickness_list,
                "Wx": Wx_list,
                "Hollow_W": hollow_W_list,
                "Theta": theta_list,
                "transmission_tensor": transmission_tensor.cpu().numpy(),
                "phase_tensor": phase_tensor.cpu().numpy()
                }
        elif params["shape_type"] == "hollow_circle":
            # R list
            R_list = np.linspace(params["R_min"], params["R_max"], params["R_n"])
            # hollow_R list
            hollow_R_list = np.linspace(params["hollow_R_min"], params["hollow_R_max"], params["hollow_R_n"])
            # 計算總迴圈數
            total_iterations = (
                len(wavelength_list) * len(period_list) * len(thickness_list) *
                len(R_list) * len(hollow_R_list)
            )

            # 依序迴圈，計算每一組 RCWA(Wavelength, Period, Thickness, Wx, Wy, theta)存入空矩陣shape為(wavelength_n, period_n, thickness_n, Wx_n, Wy_n, theta_n)
            # 創建一個 tensor，初始化為零
            transmission_tensor = torch.zeros(len(wavelength_list), len(period_list), len(thickness_list), len(R_list), len(hollow_R_list), 8)
            phase_tensor = torch.zeros_like(transmission_tensor)
            # 進行批次計算
            for i, wavelength in enumerate(wavelength_list):
                for j, period in enumerate(period_list):
                    for k, thickness in enumerate(thickness_list):
                        for l, R in enumerate(R_list):
                            for m, hollow_R in enumerate(hollow_R_list):
                                while self.is_paused:
                                        QApplication.processEvents()
                                if not self.is_running:
                                    return
                                # 模擬 RCWA 計算
                                rcwa_obj = RCWA(
                                    device=params["device"],
                                    shape_type=params["shape_type"],
                                    harmonic_order=params["harmonic_order"],
                                    wavelength=wavelength,
                                    period=period,
                                    substrate_material=params["substrate_material"],
                                    slab_material=params["slab_material"],
                                    slab_thickness=params["slab_thickness"],
                                    metasurface_material=params["metasurface_material"],
                                    metasurface_thickness=thickness,
                                    filling_material=params["filling_material"],
                                    filling_thickness=params["filling_thickness"],
                                    output_material=params["output_material"],
                                    R=R,
                                    hollow_R=hollow_R,
                                )
                                txx, txy, tyx, tyy = rcwa_obj.get_Sparameter()
                                transmission_xx = torch.abs(txx)**2
                                transmission_yy = torch.abs(tyy)**2
                                phase_xx = torch.angle(txx)
                                phase_yy = torch.angle(tyy)
                                transmission_xy = torch.abs(txy)**2
                                transmission_yx = torch.abs(tyx)**2
                                phase_xy = torch.angle(txy)
                                phase_yx = torch.angle(tyx)
                                tRL = 0.5*((txx - tyy) - 1j * (txy + tyx))
                                tRR = 0.5*((txx + tyy) + 1j * (txy - tyx))
                                tLR = 0.5*((txx - tyy) + 1j * (txy + tyx))
                                tLL = 0.5*((txx + tyy) - 1j * (txy - tyx))
                                transmission_RL = torch.abs(tRL)**2
                                transmission_LR = torch.abs(tLR)**2
                                phase_RL = torch.angle(tRL)
                                phase_LR = torch.angle(tLR)
                                transmission_RR = torch.abs(tRR)**2
                                transmission_LL = torch.abs(tLL)**2
                                phase_RR = torch.angle(tRR)
                                phase_LL = torch.angle(tLL)

                                phase_tensor[i, j, k, l, m, 0] = phase_xx
                                phase_tensor[i, j, k, l, m, 1] = phase_yx
                                phase_tensor[i, j, k, l, m, 2] = phase_xy
                                phase_tensor[i, j, k, l, m, 3] = phase_yy
                                phase_tensor[i, j, k, l, m, 4] = phase_LL
                                phase_tensor[i, j, k, l, m, 5] = phase_RL
                                phase_tensor[i, j, k, l, m, 6] = phase_LR
                                phase_tensor[i, j, k, l, m, 7] = phase_RR
                                transmission_tensor[i, j, k, l, m, 0] = transmission_xx
                                transmission_tensor[i, j, k, l, m, 1] = transmission_yx
                                transmission_tensor[i, j, k, l, m, 2] = transmission_xy
                                transmission_tensor[i, j, k, l, m, 3] = transmission_yy
                                transmission_tensor[i, j, k, l, m, 4] = transmission_LL
                                transmission_tensor[i, j, k, l, m, 5] = transmission_RL
                                transmission_tensor[i, j, k, l, m, 6] = transmission_LR
                                transmission_tensor[i, j, k, l, m, 7] = transmission_RR

                                

                                # 更新進度條
                                current_iteration += 1
                                progress = int((current_iteration / total_iterations) * 100)
                                self.progress_bar.setValue(progress)
                                
                                # 允許 UI 更新
                                QApplication.processEvents()
            self.data_sheet = {
                "shape_type": params["shape_type"],
                "Wavelength": wavelength_list,
                "Period": period_list,
                "Thickness": thickness_list,
                "R": R_list,
                "Hollow_R": hollow_R_list,
                "transmission_tensor": transmission_tensor.cpu().numpy(),
                "phase_tensor": phase_tensor.cpu().numpy()
            }
        self.is_running = False
        self.is_paused = False
        self.batch_button.setText("batch calculate")
            

    def save_tensors_to_file(self):
        """
        使用者選擇檔案格式、位置與檔名，將 Tensors 保存為 .mat、.npy 或 .txt 檔
        """
        # 彈出檔案儲存對話框，允許選擇 .mat、.npy 或 .txt 格式
        options = QFileDialog.Options()
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            "NumPy file (*.npy)",
            options=options
        )

        if file_path:
            # 判斷選擇的檔案格

            if selected_filter == "NumPy file (*.npy)" or file_path.endswith(".npy"):
                # 保存為 .npy 檔
                np.save(file_path, {
                    "data_sheet": self.data_sheet
                })
                print(f"Tensors saved as NumPy file to {file_path}")

                # 修改路徑後綴為 .mat
                mat_file_path = file_path[:-4] + ".mat"

                # 保存為 .mat 檔
                sio.savemat(mat_file_path, {
                    "data_sheet": self.data_sheet
                })
                print(f"Tensors also saved as MAT file to {mat_file_path}")
                

    def openDataVisualizer(self):
        """
        按下按鈕後，依照 self.data_sheet 是否為 None 來決定要怎麼開 DataVisualizer。
        """
        if self.data_sheet is not None:
            # 情況一：直接將 data_sheet 傳給 DataVisualizer
            self.vis_window = DataVisualize(data_sheet=self.data_sheet)
        else:
            # 情況二：data_sheet 還沒準備好，讓 DataVisualizer 自己顯示「讀取檔案」按鈕
            self.vis_window = DataVisualize()
        
        self.vis_window.show()

def main():
    #stored_hardware_id = '95a5f4774e6a69f2079b145c9e52c28ede474b7a4bbd07a5f7d606b199d1f006'
    
    # Check hardware ID
    """ current_hardware_id = get_hardware_id()
    #print(current_hardware_id)
    if stored_hardware_id not in current_hardware_id:
        print("此應用程式只能在授權的設備上執行")
        sys.exit() """

    # Check expiry date
    expiry_date = read_expiry_date_from_json('expiry_date.json')
    current_date = get_network_time()

    if current_date is None:
        print("無法驗證當前日期，請檢察網路")
        sys.exit()

    if current_date > expiry_date:
        print("該用戶已過期")
        sys.exit()
    app = QApplication(sys.argv)
    
    # 建立 QFont 物件並指定字型、大小
    font = QFont("Arial", 14)  # 這裡指定字型為 "Arial"，大小為 14pt
    app.setFont(font)
    window = MainWindow()
    window.setGeometry(200, 100, 800, 600)  
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
