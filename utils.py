from PyQt5.QtWidgets import QComboBox, QFormLayout, QLineEdit

def create_input_field(self, label_text, default_value, parent_group=None):
        input_field = QLineEdit(default_value)
        self.input_fields[label_text] = input_field
        if parent_group:
            layout = parent_group.layout() if parent_group.layout() else QFormLayout()
            layout.addRow(label_text + ":", input_field)
            parent_group.setLayout(layout)
        return input_field

def create_combo_box(self, label_text, items):
    combo_box = QComboBox()
    combo_box.addItems(items)
    combo_box.setCurrentIndex(0)
    self.combo_boxes[label_text] = combo_box
    return combo_box