# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import sys
from PySide6 import QtWidgets
from PySide6.QtGui import QStandardItemModel, QStandardItem

from VeraGrid.Gui.gui_functions import get_icon_list_model
from VeraGrid.Gui.RmsModelEditor.rms_model_editor import Ui_MainWindow
from VeraGrid.Gui.RmsModelEditor.block_editor import BlockEditor
from VeraGridEngine.Devices.Dynamic.dynamic_model_host import DynamicModelHost
import VeraGridEngine.Devices as dev

class RmsModelEditorGUI(QtWidgets.QMainWindow):

    def __init__(self, model_host: DynamicModelHost, parent=None, ):
        """

        :param parent:
        """
        QtWidgets.QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('RMS Model editor')

        self.model_host: DynamicModelHost = model_host
        self.editor = BlockEditor()
        self.ui.editorLayout.addWidget(self.editor)

        # Table model for variables/equations (right side)
        self._data_table_model = QStandardItemModel()
        self.ui.datatableView.setModel(self._data_table_model)

        # Connect category selection to table update
        self.ui.datalistWidget.itemSelectionChanged.connect(self.update_table)


        # self.ui.actionCheckModel.triggered.connect(self.extract_dae)

    @property
    def model(self):
        return self.editor.block_system


    def update_table(self):
        items = self.ui.datalistWidget.selectedItems()
        if not items:
            return

        category = items[0].text()
        self._data_table_model.clear()
        self._data_table_model.setHorizontalHeaderLabels(["Name", "Type", "Equation"])

        if category == "State variables":
            for submodel in self.model.children:
                for var, eq in zip(submodel.state_vars, submodel.state_eqs):
                    self._data_table_model.appendRow([
                        QStandardItem(var.name),
                        QStandardItem("state"),
                        QStandardItem(str(eq)),
                ])

        elif category == "Algebraic variables":
            for submodel in self.model.children:
                for var, eq in zip(submodel.algebraic_vars, submodel.algebraic_eqs):
                    self._data_table_model.appendRow([
                        QStandardItem(var.name),
                        QStandardItem("algebraic"),
                        QStandardItem(str(eq)),
                ])

        elif category == "Constants":
            for submodel in self.model.children:
                for const in submodel.parameters:
                    self._data_table_model.appendRow([
                        QStandardItem(const.name),
                        QStandardItem("parameter"),
                        QStandardItem(str(const.value)),
                ])

    def extract_dae(self):
        eqs = self.editor.run()

        for eq in eqs:
            print(str(eq))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    md = dev.DynamicModelHost()
    window = RmsModelEditorGUI(md)
    window.resize(1.61 * 700.0, 600.0)  # golden ratio
    window.show()
    sys.exit(app.exec())
