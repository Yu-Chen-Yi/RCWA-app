import sys
import torch
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget,
    QLabel, QSlider, QComboBox, QHBoxLayout, QFileDialog, QLineEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


class DataVisualize(QWidget):
    """
    獨立新視窗用於顯示 Matplotlib 圖並提供多維切片選擇功能，
    同時繪製 Transmission & Phase (1D or 2D)。
    """
    def __init__(
        self,
        data_sheet=None,
        default_colormap="jet",
        T_clim = [0, 1],
        phase_clim = [0, 2*np.pi]
    ):
        super().__init__()
        self.setWindowIcon(QIcon('your_icon.ico'))  # 圖示檔案放在同一個資料夾內
        if data_sheet is None:
            # 如果取消這裡data_sheet會是None，會導致後面的parseDataSheet出錯
            data_sheet = self.load_npy_file()
        self.setWindowTitle("Data Visualizer")
        self.setWindowFlag(Qt.Window)  # 設定為獨立視窗
        self.T_clim = T_clim
        self.phase_clim = phase_clim

        if data_sheet is not None:
            self.parseDataSheet(data_sheet)
            
            self.num_dims = len(self.dimension_names)
            # 用來記錄「每個維度」目前 slice 的索引
            
            self.slice_indices = [0] * (self.num_dims+1)

            # ========== 1. 建立 GUI 元件 ==========
            # 創建一個load npy file的按鈕
            self.load_npy_button = QPushButton("Load NPY File")
            self.load_npy_button.clicked.connect(self.openDataVisualizer)

            # (A) 選擇 x、y 維度的 ComboBox
            self.combo_x_dim = QComboBox()
            self.combo_y_dim = QComboBox()
            self.combo_x_dim.addItems(self.dimension_names)
            self.combo_y_dim.addItems(self.dimension_names)

            # 預設先選第0維當 x， 若維度>1，則第1維當 y
            self.combo_x_dim.setCurrentIndex(0)
            self.combo_y_dim.setCurrentIndex(1)

            self.combo_x_dim.currentIndexChanged.connect(self.on_dim_combo_changed)
            self.combo_y_dim.currentIndexChanged.connect(self.on_dim_combo_changed)

            # (A') 選擇 colormap 的 ComboBox
            self.combo_colormap = QComboBox()
            for cmap in plt.colormaps():
                self.combo_colormap.addItem(cmap)
            self.combo_colormap.setCurrentText(default_colormap)
            self.combo_colormap.currentIndexChanged.connect(self.on_dim_combo_changed)

            self.combo_polarization = QComboBox()
            self.combo_polarization.addItems(["XLP->XLP", "XLP->YLP", "YLP->XLP", "YLP->YLP", "LCP->LCP", "LCP->RCP", "RCP->LCP", "RCP->RCP"])
            self.combo_polarization.setCurrentIndex(0)
            self.combo_polarization.currentIndexChanged.connect(self.chosen_polarization)
            # (B) 建立 Matplotlib 圖 (含兩個子圖)
            self.figure = Figure()
            self.ax_transmission = self.figure.add_subplot(121)  # 左子圖：Transmission
            self.ax_phase = self.figure.add_subplot(122)         # 右子圖：Phase
            self.canvas = FigureCanvas(self.figure)

            # (C) 建立「每個維度」對應的 Slider + Label
            self.sliders = []
            self.slider_labels = []

            sliders_layout = QVBoxLayout()
            for dim in range(self.num_dims):
                label = QLabel(f"{self.dimension_names[dim]} : {self.dimension_list[dim][0]:.2f} nm, 第1/{len(self.dimension_list[dim])}筆")
                slider = QSlider(Qt.Horizontal)
                slider.setMinimum(0)
                slider.setMaximum(self.data1.shape[dim] - 1)
                slider.setValue(0)

                # 用 lambda + 預先捕捉 dim
                slider.valueChanged.connect(lambda value, d=dim: self.update_slice(d, value))

                self.sliders.append(slider)
                self.slider_labels.append(label)
                
                sliders_layout.addWidget(label)
                sliders_layout.addWidget(slider)

            # (D) 上方放 ComboBox (x_dim, y_dim, colormap)，下面放 sliders + 畫布
            combo_layout = QHBoxLayout()
            X_dimension_layout = QHBoxLayout()
            Y_dimension_layout = QHBoxLayout()
            Colormap_layout = QHBoxLayout()
            Polarization_layout = QHBoxLayout()
            X_dimension_layout.addWidget(QLabel("X dimension:"))
            X_dimension_layout.addWidget(self.combo_x_dim)
            Y_dimension_layout.addWidget(QLabel("Y dimension:"))
            Y_dimension_layout.addWidget(self.combo_y_dim)
            Colormap_layout.addWidget(QLabel("Colormap:"))
            Colormap_layout.addWidget(self.combo_colormap)
            Polarization_layout.addWidget(QLabel("Polarization:"))
            Polarization_layout.addWidget(self.combo_polarization)
            combo_layout.addLayout(X_dimension_layout)
            combo_layout.addLayout(Y_dimension_layout)
            combo_layout.addLayout(Colormap_layout)
            combo_layout.addLayout(Polarization_layout)
            
            
            clim_layout = QHBoxLayout()
            # colorbar limit layout
            transmittance_clim_label = QLabel("Transmittance color scale:")
            self.transmittance_min_input = QLineEdit()
            self.transmittance_max_input = QLineEdit()
            self.transmittance_min_input.setText("0")
            self.transmittance_max_input.setText("1")
            self.transmittance_min_input.textChanged.connect(self.clim_changed)
            self.transmittance_max_input.textChanged.connect(self.clim_changed)
            clim_layout.addWidget(transmittance_clim_label)
            clim_layout.addWidget(self.transmittance_min_input)
            clim_layout.addWidget(self.transmittance_max_input)
        
            phase_clim_label = QLabel("Phase color scale:")
            self.phase_min_input = QLineEdit()
            self.phase_max_input = QLineEdit()
            self.phase_min_input.setText("0")
            self.phase_max_input.setText("6.283")
            self.phase_min_input.textChanged.connect(self.clim_changed)
            self.phase_max_input.textChanged.connect(self.clim_changed)
            clim_layout.addWidget(phase_clim_label)
            clim_layout.addWidget(self.phase_min_input)
            clim_layout.addWidget(self.phase_max_input)
            

            main_layout = QVBoxLayout()
            main_layout.addWidget(self.load_npy_button)
            main_layout.addLayout(combo_layout)
            main_layout.addLayout(sliders_layout)
            main_layout.addLayout(clim_layout)
            main_layout.addWidget(self.canvas)

            self.setLayout(main_layout)

            # 第一次進來先更新一次
            self.on_dim_combo_changed()

    def on_dim_combo_changed(self):
        """
        當 x or y 維度或 colormap 被改變時，需要：
          1. 禁用被選為 x,y 的 Slider
          2. 啟用其他 Slider
          3. 更新繪圖
        """
        x_dim = self.combo_x_dim.currentIndex()
        y_dim = self.combo_y_dim.currentIndex()

        for dim in range(self.num_dims):
            if dim == x_dim or dim == y_dim:
                self.sliders[dim].setDisabled(True)
            else:
                self.sliders[dim].setDisabled(False)

        self.update_plot()

    def clim_changed(self):
        """
        當 x or y 維度或 colormap 被改變時，需要：
          1. 禁用被選為 x,y 的 Slider
          2. 啟用其他 Slider
          3. 更新繪圖
        """
        
        try:
            t_min = float(self.transmittance_min_input.text())  if self.transmittance_min_input.text() else 0.0
        except (ValueError, TypeError):
            t_min = 0.0

        try:
            t_max = float(self.transmittance_max_input.text())  if self.transmittance_max_input.text() else 1.0
        except (ValueError, TypeError):
            t_max = 1.0

        try:
            p_min = float(self.phase_min_input.text())  if self.phase_min_input.text() else 0.0
        except (ValueError, TypeError):
            # 轉換失敗時回傳預設值 0.0
            p_min = 0.0
        try:
            # 嘗試轉換為浮點數並使用 floor 檢查合法性
            p_max = float(self.phase_max_input.text())  if self.phase_max_input.text() else (2*np.pi)
        except (ValueError, TypeError):
            p_max = np.pi*2

        if t_max < t_min:
            return
        if p_max < p_min:
            return
        self.T_clim = [t_min, t_max]
        self.phase_clim = [p_min, p_max]

        self.update_plot()

    def update_slice(self, dim, value):
        """更新指定維度的 slice 索引"""
        self.slice_indices[dim] = value
        self.slider_labels[dim].setText(f"{self.dimension_names[dim]} : {self.dimension_list[dim][value]:.2f} nm, 第{value+1}/{len(self.dimension_list[dim])}筆")
        self.update_plot()

    def chosen_polarization(self, value):
        """
        當使用者選擇不同的 polarization 時，更新並重繪。
        """
        self.slice_indices[-1] = value
        self.update_plot()

    def update_plot(self):
        """
        依照使用者目前設定的 x_dim, y_dim，以及剩餘維度的 slice 索引，
        同時繪製 Transmission & Phase。維持以下邏輯：
          - 如果 x_dim == y_dim，就當作 1D 來畫線圖
          - 如果 x_dim != y_dim，就當作 2D 來用 imshow
          - 做 phase 修正 (將第一個點對齊到 0，再 mod 2π)
        """

        # 清除舊圖
        self.ax_transmission.clear()
        self.ax_phase.clear()
        
        # 移除舊的 colorbar
        if self.colorbar1 is not None:
            self.colorbar1.ax.remove()
            self.colorbar1 = None
        if self.colorbar2 is not None:
            self.colorbar2.ax.remove()
            self.colorbar2 = None

        x_dim = self.combo_x_dim.currentIndex()
        y_dim = self.combo_y_dim.currentIndex()
        colormap_name = self.combo_colormap.currentText()

        # 構造 slicing
        slicing = []
        for dim in range(self.num_dims+1):
            if dim == x_dim or dim == y_dim:
                slicing.append(slice(None))
            else:
                slicing.append(self.slice_indices[dim])

        # 清除子圖
        self.ax_transmission.clear()
        self.ax_phase.clear()

        # 取得切片資料
        data_trans = self.data1[tuple(slicing)]
        data_phase = self.data2[tuple(slicing)]

        # =========== 1D / 2D 判斷 =========== #
        if x_dim == y_dim:
            # 1D 繪圖
            data_trans_1d = np.array(data_trans).ravel()
            data_phase_1d = np.array(data_phase).ravel()

            # 做 phase 修正：讓第一點成 0 並轉到 0~2π
            if len(data_phase_1d) > 0:
                data_phase_1d = np.mod(data_phase_1d - data_phase_1d[0], 2*np.pi)

            # 畫 Transmission
            x_data = self.dimension_list[x_dim]
            self.ax_transmission.plot(x_data, data_trans_1d, marker='o', linestyle='-')
            #self.ax_transmission.set_title("Transmittance (a.u.)")
            self.ax_transmission.set_xlabel(self.dimension_names[x_dim])
            self.ax_transmission.set_ylabel("Transmittance (a.u.)")
            self.ax_transmission.set_ylim(self.T_clim)  # 假設在 0~1

            # 畫 Phase
            self.ax_phase.plot(x_data, data_phase_1d, marker='o', linestyle='-', color='r')
            #self.ax_phase.set_title("Phase (1D)")
            self.ax_phase.set_xlabel(self.dimension_names[x_dim])
            self.ax_phase.set_ylabel("Phase (rad)")
            self.ax_phase.set_ylim(self.phase_clim) 
        else:
            # 2D 繪圖
            data_trans_2d = np.array(data_trans)
            data_phase_2d = np.array(data_phase)

            # phase 修正
            if data_phase_2d.size > 0:
                data_phase_2d = np.mod(data_phase_2d - data_phase_2d.flat[0], 2*np.pi)

            # 若要確保 x_dim 對應水平軸、y_dim 對應垂直軸，可根據 x_dim < y_dim 來 transpose
            if x_dim < y_dim:
                data_trans_2d = data_trans_2d.T
                data_phase_2d = data_phase_2d.T
            
            # x, y 軸資訊
            extent = [
                self.dimension_list[x_dim][0],
                self.dimension_list[x_dim][-1],
                self.dimension_list[y_dim][0],
                self.dimension_list[y_dim][-1],
            ]

            # Transmission
            im1 = self.ax_transmission.imshow(
                data_trans_2d,
                extent=extent,
                cmap=colormap_name,
                origin='lower',
                aspect='auto',
                vmin=self.T_clim[0], vmax=self.T_clim[-1]
            )
            self.ax_transmission.set_title("Transmittance (a.u.)")
            self.ax_transmission.set_xlabel(self.dimension_names[x_dim])
            self.ax_transmission.set_ylabel(self.dimension_names[y_dim])

            # 動態加 colorbar
            divider1 = make_axes_locatable(self.ax_transmission)
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            self.colorbar1 = self.figure.colorbar(im1, cax=cax1)

            # Phase
            im2 = self.ax_phase.imshow(
                data_phase_2d,
                extent=extent,
                cmap=colormap_name,
                origin='lower',
                aspect='auto',
                vmin=self.phase_clim[0], vmax=self.phase_clim[-1]
            )
            self.ax_phase.set_title("Phase (rad)")
            self.ax_phase.set_xlabel(self.dimension_names[x_dim])
            self.ax_phase.set_ylabel(self.dimension_names[y_dim])

            divider2 = make_axes_locatable(self.ax_phase)
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            self.colorbar2 = self.figure.colorbar(im2, cax=cax2)

        # Figure 大標題顯示 slice 狀態
        """ self.figure.suptitle(
            [f"{self.dimension_names[d][:-5]}={self.dimension_list[d][i]} nm" for d, i in enumerate(self.slice_indices)
             if d != x_dim and d != y_dim],
            fontsize=10
        ) """
        # 重繪
        self.canvas.draw()

    def parseDataSheet(self, data_sheet):
        """
        依照 shape_type（rectangle, ellipse, circle）解析維度資訊。
        """
        shape_type = data_sheet.get("shape_type", "unknown")
        
        wavelength_list = data_sheet["Wavelength"]
        period_list = data_sheet["Period"]
        thickness_list = data_sheet["Thickness"]
        
        if shape_type == 'rectangle' or shape_type == 'rhombus' or shape_type == 'cross':
            dimension_names = [
                "Wavelength (nm)", "Period (nm)", "Thickness (nm)",
                "Wx (nm)", "Wy (nm)", "Rotation Angle (deg)"
            ]
            width_list = data_sheet["Wx"]
            height_list = data_sheet["Wy"]
            rotation_angle_list = data_sheet["Theta"]
            dimension_list = [
                wavelength_list, period_list, thickness_list,
                width_list, height_list, rotation_angle_list
            ]
        elif shape_type == 'ellipse':
            dimension_names = [
                "Wavelength (nm)", "Period (nm)", "Thickness (nm)",
                "Rx (nm)", "Ry (nm)", "Rotation Angle (deg)"
            ]
            width_list = data_sheet["Rx"]
            height_list = data_sheet["Ry"]
            rotation_angle_list = data_sheet["Theta"]
            dimension_list = [
                wavelength_list, period_list, thickness_list,
                width_list, height_list, rotation_angle_list
            ]
        elif shape_type == 'circle':
            dimension_names = [
                "Wavelength (nm)", "Period (nm)", "Thickness (nm)",
                "R (nm)"
            ]
            r_list = data_sheet["R"]
            dimension_list = [
                wavelength_list, period_list, thickness_list, r_list
            ]
        elif shape_type == 'square':
            dimension_names = [
                "Wavelength (nm)", "Period (nm)", "Thickness (nm)",
                "W (nm)", "Rotation Angle (deg)"
            ]
            width_list = data_sheet["Wx"]
            rotation_angle_list = data_sheet["Theta"]
            dimension_list = [
                wavelength_list, period_list, thickness_list,
                width_list, rotation_angle_list
            ]
        elif shape_type == 'hollow_square':
            dimension_names = [
                "Wavelength (nm)", "Period (nm)", "Thickness (nm)",
                "W (nm)", "Hollow Width (nm)", "Rotation Angle (deg)"
            ]
            width_list = data_sheet["Wx"]
            hollow_width_list = data_sheet["Hollow_W"]
            rotation_angle_list = data_sheet["Theta"]
            dimension_list = [
                wavelength_list, period_list, thickness_list,
                width_list, hollow_width_list, rotation_angle_list
            ]
        elif shape_type == 'hollow_circle':
            dimension_names = [
                "Wavelength (nm)", "Period (nm)", "Thickness (nm)",
                "R (nm)", "Hollow R (nm)"
            ]
            r_list = data_sheet["R"]
            hollow_width_list = data_sheet["Hollow_R"]
            dimension_list = [
                wavelength_list, period_list, thickness_list,
                r_list, hollow_width_list
            ]
        else:
            raise ValueError(f"shape_type not recognized: {shape_type}")
        
        # 取出傳輸與相位資料
        self.data1 = data_sheet["transmission_tensor"]
        self.data2 = data_sheet["phase_tensor"]
        
        # 儲存維度資訊
        self.dimension_names = dimension_names
        self.dimension_list = dimension_list
        
        # 預設 x-dim, y-dim, slice-dim
        self.x_dim_index = 0
        self.y_dim_index = 1
        self.slice_dim_index = 2
        # colorbar 參考
        self.colorbar1 = None
        self.colorbar2 = None

    def load_npy_file(self):
        """
        讓使用者選擇 .npy 檔，讀取後嘗試解析成 data_sheet，
        並從其中取出 transmission_tensor、phase_tensor。
        """
        filename, _ = QFileDialog.getOpenFileName(
            self, "選擇 NPY 檔案", "", "NPY Files (*.npy)"
        )
        if not filename:
            return  # 使用者取消

        try:
            data = np.load(filename, allow_pickle=True).item()
        except Exception as e:
            print(f"讀取檔案失敗: {e}")
            return

        if "data_sheet" not in data:
            print("檔案內容不符合預期，請確認其中含有 data_sheet")
            return

        data_sheet = data["data_sheet"]
        return data_sheet
    
    def openDataVisualizer(self):
        """
        按下按鈕後，依照 self.data_sheet 是否為 None 來決定要怎麼開 DataVisualizer。
        """
        # 情況二：data_sheet 還沒準備好，讓 DataVisualizer 自己顯示「讀取檔案」按鈕
        self.vis_window = DataVisualize()
        
        self.vis_window.show()