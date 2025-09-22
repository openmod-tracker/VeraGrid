# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from enum import Enum, auto
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsItem,
                               QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsTextItem, QMenu, QGraphicsPathItem,
                               QDialog, QVBoxLayout, QComboBox, QDialogButtonBox, QInputDialog, QLabel, QDoubleSpinBox)
from PySide6.QtGui import QPen, QBrush, QPainterPath, QAction, QPainter
from PySide6.QtCore import Qt, QPointF
from VeraGridEngine.Utils.Symbolic.block import (
    Block,
    adder,
    constant,
    variable,
    gain,
    integrator,
)
from VeraGridEngine.Utils.Symbolic.block_solver import BlockSolver
from VeraGridEngine.Utils.Symbolic.symbolic import Var



@dataclass
class BlockBridge:
    gui: "BlockItem"  # visual node
    outs: List[Var]  # exactly len(gui.outputs)
    ins: List[Var]  # exactly len(gui.inputs) – placeholders
    api_blocks: List[Block]  # usually length 1, but e.g. PI returns 4


class BlockType(Enum):
    GAIN = auto()
    SUM = auto()
    INTEGRATOR = auto()
    DERIVATIVE = auto()
    PRODUCT = auto()
    DIVIDE = auto()
    SQRT = auto()
    SQUARE = auto()
    ABS = auto()
    MIN = auto()
    MAX = auto()
    STEP = auto()
    CONSTANT = auto()
    VARIABLE = auto()
    SATURATION = auto()
    RELATIONAL = auto()
    LOGICAL = auto()
    SOURCE = auto()
    DRAIN = auto()
    GENERIC = auto()

class BlockTypeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Block Type")
        self.layout = QVBoxLayout(self)

        self.combo = QComboBox(self)
        for bt in BlockType:
            self.combo.addItem(bt.name, bt)
        self.layout.addWidget(self.combo)

        # 👇 Extra field for constants
        self.value_label = QLabel("Constant value:", self)
        self.value_spin = QDoubleSpinBox(self)
        self.value_spin.setRange(-1e6, 1e6)
        self.value_spin.setValue(0.0)
        self.layout.addWidget(self.value_label)
        self.layout.addWidget(self.value_spin)

        # Initially hidden
        self.value_label.hide()
        self.value_spin.hide()

        self.combo.currentIndexChanged.connect(self._on_block_changed)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def _on_block_changed(self, index):
        block_type = self.combo.itemData(index)
        if block_type == BlockType.CONSTANT:
            self.value_label.show()
            self.value_spin.show()
        else:
            self.value_label.hide()
            self.value_spin.hide()

    def selected_block_type(self) -> BlockType:
        return self.combo.currentData()

    def constant_value(self) -> float:
        return self.value_spin.value()


class PortItem(QGraphicsEllipseItem):
    """
    Port of a block
    """

    def __init__(self,
                 block: "BlockItem",
                 is_input: bool,
                 index: int, # number of inputs
                 total: int,
                 radius=6):
        """

        :param block:
        :param is_input:
        :param index:
        :param total:
        :param radius:
        """
        super().__init__(-radius, -radius, 2 * radius, 2 * radius, block)
        self.setBrush(QBrush(Qt.GlobalColor.blue if is_input else Qt.GlobalColor.green))
        self.setPen(QPen(Qt.GlobalColor.black))
        self.setZValue(1)
        self.setAcceptHoverEvents(True)
        self.block = block
        self.is_input = is_input
        self.connection = None
        self.index = index
        self.total = total

        spacing = block.rect().height() / (total + 1)
        y = spacing * (index + 1)
        x = 0 if is_input else block.rect().width()
        self.setPos(x, y)

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def is_connected(self):
        return self.connection is not None


class ConnectionItem(QGraphicsPathItem):
    def __init__(self, source_port, target_port):
        super().__init__()
        self.setZValue(-1)
        self.source_port = source_port
        self.target_port = target_port
        self.source_port.connection = self
        self.target_port.connection = self
        self.setPen(QPen(Qt.GlobalColor.darkBlue, 2))
        self.setAcceptHoverEvents(True)

        self.update_path()

    def update_path(self):
        start = self.source_port.scenePos()
        end = self.target_port.scenePos()
        mid_x = (start.x() + end.x()) / 2
        c1 = QPointF(mid_x, start.y())
        c2 = QPointF(mid_x, end.y())
        path = QPainterPath(start)
        path.cubicTo(c1, c2, end)
        self.setPath(path)

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CursorShape.PointingHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def contextMenuEvent(self, event):
        menu = QMenu()
        remove_action = QAction("Remove Connection", menu)
        menu.addAction(remove_action)
        if menu.exec(event.screenPos()) == remove_action:
            self.scene().removeItem(self)
            self.source_port.connection = None
            self.target_port.connection = None


class ResizeHandle(QGraphicsRectItem):
    def __init__(self, block, size=10):
        super().__init__(0, 0, size, size, block)
        self.setBrush(QBrush(Qt.GlobalColor.darkGray))
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self.setZValue(2)
        self.block = block
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        self.setAcceptHoverEvents(True)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if not self.block._resizing_from_handle:
                return super().itemChange(change, value)

            new_pos = value  # already QPointF
            min_width, min_height = 40, 30
            new_width = max(new_pos.x(), min_width)
            new_height = max(new_pos.y(), min_height)

            self.block.resize_block(new_width, new_height)

            return QPointF(new_width, new_height)
        return super().itemChange(change, value)


class BlockItem(QGraphicsRectItem):
    def __init__(self, block_sys: Block):
        """

        :param block_sys: Block
        """
        super().__init__(0, 0, 100, 60)

        # ------------------------
        # API
        # ------------------------
        self.subsys = block_sys  # << NEW

        # ---------------------------
        # Graphical stuff
        # ---------------------------
        self.setBrush(Qt.GlobalColor.lightGray)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges
        )
        self.setAcceptHoverEvents(True)
        self.setAcceptHoverEvents(True)

        self.name_item = QGraphicsTextItem(self.subsys.name, self)

        self.name_item.setPos(10, 5)

        n_inputs = len(self.subsys.in_vars)
        n_outputs = len(self.subsys.out_vars)

        self.inputs = [PortItem(self, True, i, n_inputs) for i in range(n_inputs)]
        self.outputs = [PortItem(self, False, i, n_outputs) for i in range(n_outputs)]

        self.resize_handle = ResizeHandle(self)

        # ✅ Avoid triggering overridden setRect during init
        super().setRect(0, 0, 100, 60)
        self.update_ports()
        self.update_handle_position()

        self._resizing_from_handle = False


    def resize_block(self, width, height):
        # Update geometry safely
        self.prepareGeometryChange()
        QGraphicsRectItem.setRect(self, 0, 0, width, height)
        self.update_ports()
        self.update_handle_position()

    def update_handle_position(self):
        rect = self.rect()
        self._resizing_from_handle = False
        self.resize_handle.setPos(rect.width(), rect.height())
        self._resizing_from_handle = True

    def _set_rect_internal(self, w, h):
        QGraphicsRectItem.setRect(self, 0, 0, w, h)
        self.update_ports()
        self.update_handle_position()

    def setRect(self, x, y, w, h):
        if not getattr(self, '_suppress_resize', False):
            self._set_rect_internal(w, h)

    def update_ports(self):
        for i, port in enumerate(self.inputs):
            spacing = self.rect().height() / (len(self.inputs) + 1)
            port.setPos(0, spacing * (i + 1))
        for i, port in enumerate(self.outputs):
            spacing = self.rect().height() / (len(self.outputs) + 1)
            port.setPos(self.rect().width(), spacing * (i + 1))
        self.update_handle_position()
        # Also update connections
        for port in self.inputs + self.outputs:
            if port.connection:
                port.connection.update_path()

    def hoverEnterEvent(self, event):
        QApplication.setOverrideCursor(Qt.CursorShape.OpenHandCursor)

    def hoverLeaveEvent(self, event):
        QApplication.restoreOverrideCursor()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            for port in self.inputs + self.outputs:
                if port.connection:
                    port.connection.update_path()
        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        menu = QMenu()
        delete_action = QAction("Remove Block", menu)
        menu.addAction(delete_action)
        if menu.exec(event.screenPos()) == delete_action:
            # Remove connections
            for port in self.inputs + self.outputs:
                if port.connection:
                    self.scene().removeItem(port.connection)
                    if port.connection.source_port:
                        port.connection.source_port.connection = None
                    if port.connection.target_port:
                        port.connection.target_port.connection = None
            # Remove the block itself
            self.scene().removeItem(self)


class GraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(self.renderHints() | QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self._panning = False
        self._pan_start = QPointF()

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = 1.15 if zoom_in else 1 / 1.15
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - delta.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - delta.y()))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)


def create_block_of_type(block_type: BlockType, ins: int, outs: int, const_value: Optional[float] = None) -> Block:
    """
    Create a Block appropriate for block_type. Use placeholder Vars for inputs/outputs
    so that Block.in_vars and Block.out_vars are not empty.
    """

    def placeholders(n, base):
        return [Var(f"{base}{i}") for i in range(n)]

    if block_type == BlockType.CONSTANT:
        value = const_value if const_value is not None else 0.0
        y, blk = constant(value, name="const")
        blk.in_vars = []
        blk.out_vars = [y]
        return blk

    if block_type == BlockType.VARIABLE:
        y, blk = variable(name="variable")
        blk.in_vars = []
        blk.out_vars = [y]
        return blk

    # GAIN (single input -> single output)
    if block_type == BlockType.GAIN:
        u = Var("gain_u")  # placeholder input var
        y, blk = gain(1.0, u, name="gain_out")
        if not getattr(blk, "in_vars", None):
            blk.in_vars = [u]
        if not getattr(blk, "out_vars", None):
            blk.out_vars = [y]
        return blk

    # SUM / ADDER (N inputs)
    if block_type == BlockType.SUM or block_type == BlockType.PRODUCT:
        # for SUM use adder; for PRODUCT you may later implement product()
        inputs = placeholders(ins, "sum_in_")
        y, blk = adder(inputs, name="sum_out")
        if not getattr(blk, "in_vars", None):
            blk.in_vars = inputs
        if not getattr(blk, "out_vars", None):
            blk.out_vars = [y]
        return blk

    # INTEGRATOR (1 input -> 1 state output)
    if block_type == BlockType.INTEGRATOR:
        u = Var("int_u")
        x, blk = integrator(u, name="x")
        if not getattr(blk, "in_vars", None):
            blk.in_vars = [u]
        if not getattr(blk, "out_vars", None):
            blk.out_vars = [x]
        return blk

    # SOURCE: a block with only an output (like a constant/source)
    if block_type == BlockType.SOURCE:
        y, blk = constant(0.0, name="source_out")
        if not getattr(blk, "out_vars", None):
            blk.out_vars = [y]
        blk.in_vars = []
        return blk

    # DRAIN: a sink with inputs but no outputs
    if block_type == BlockType.DRAIN:
        ins_vars = placeholders(ins, "drain_in_")
        blk = Block(name="DRAIN")
        blk.in_vars = ins_vars
        blk.out_vars = []
        return blk

    # GENERIC / fallback: create a block and attach placeholder vars
    in_vars = placeholders(ins, f"{block_type.name.lower()}_in_")
    out_vars = [Var(f"{block_type.name.lower()}_out{i}") for i in range(outs)]
    blk = Block(name=block_type.name)
    blk.in_vars = in_vars
    blk.out_vars = out_vars
    return blk


class DiagramScene(QGraphicsScene):
    def __init__(self, editor):
        super().__init__()
        self.editor = editor
        self.temp_line = None
        self.source_port = None

        self._main_block = Block()

    def get_main_block(self):
        return self._main_block

    def contextMenuEvent(self, event):
        item = self.itemAt(event.scenePos(), self.views()[0].transform())

        # Let lines, block and ports handle their own context menus
        if item is not None:
            if not isinstance(item, DiagramScene):
                return super().contextMenuEvent(event)

        dialog = BlockTypeDialog()
        if dialog.exec():
            block_type = dialog.selected_block_type()
            ins = 2 if block_type in {BlockType.SUM, BlockType.PRODUCT, BlockType.MIN, BlockType.MAX} else 1
            outs = 1
            if block_type == BlockType.SOURCE:
                ins = 0
            elif block_type == BlockType.DRAIN:
                outs = 0

            value = dialog.constant_value() if block_type == BlockType.CONSTANT else None
            self.add_block(event.scenePos(), ins, outs, block_type, value)

        return None

    def add_block(self, pos, ins, outs, block_type, const_value=None):
        blk = create_block_of_type(block_type, ins, outs, const_value)
        block_item = BlockItem(blk)

        # add to the model and the scene
        self._main_block.add(blk)
        self.addItem(block_item)

        block_item.setPos(pos)

    def mousePressEvent(self, event):
        for item in self.items(event.scenePos()):
            if isinstance(item, PortItem) and not item.is_input and not item.is_connected():
                self.source_port = item
                path = QPainterPath(item.scenePos())
                self.temp_line = self.addPath(path, QPen(Qt.PenStyle.DashLine))
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.temp_line:
            start = self.source_port.scenePos()
            end = event.scenePos()
            mid_x = (start.x() + end.x()) / 2
            c1 = QPointF(mid_x, start.y())
            c2 = QPointF(mid_x, end.y())
            path = QPainterPath(start)
            path.cubicTo(c1, c2, end)
            self.temp_line.setPath(path)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.temp_line:
            # FIX: scan items under mouse for a valid input Port
            for item in self.items(event.scenePos()):
                if isinstance(item, PortItem) and item.is_input and not item.is_connected():
                    dst_port: PortItem = item
                    connection = ConnectionItem(self.source_port, dst_port)
                    src_var = self.source_port.block.subsys.out_vars[self.source_port.index]
                    dst_var = dst_port.block.subsys.in_vars[dst_port.index]
                    dst_port.block.subsys.in_vars[dst_port.index] = self.source_port.block.subsys.out_vars[self.source_port.index]
                    self.addItem(connection)
                    break
            self.removeItem(self.temp_line)
            self.temp_line = None
            self.source_port = None
        else:
            super().mouseReleaseEvent(event)


class BlockEditor(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Block Editor with Ports")

        self.scene = DiagramScene(self)
        self.view = GraphicsView(self.scene)
        self.setCentralWidget(self.view)

        self.block_system = self.scene.get_main_block()

        self.resize(800, 600)

    # def run(self):
    #     engine = BlockSolver(block_system=self.block_system)
    #     engine.simulate(
    #         t0=0,
    #         t_end=10,
    #         h=0.01,
    #         x0=engine.get_dummy_x0(),
    #         method="implicit_euler"
    #     )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BlockEditor()
    window.show()
    sys.exit(app.exec())
