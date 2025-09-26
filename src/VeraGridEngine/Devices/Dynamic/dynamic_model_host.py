# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from typing import List, Dict, Any, Union
from dataclasses import dataclass
from VeraGridEngine.Devices.Parents.editable_device import EditableDevice
from VeraGridEngine.Utils.Symbolic.block import Block
from VeraGridEngine.Devices.Dynamic.rms_template import RmsModelTemplate
from VeraGridEngine.enumerations import DeviceType


@dataclass
# class BlockDiagramNode:
#     x: float
#     y: float
#     tpe: str
#     device_uid: int
#
#     def get_node_dict(self):
#         """
#         get as a dictionary point
#         :return:
#         """
#         return {'x': self.x,
#                 'y': self.y,
#                 'type': self.tpe,
#                 'device_uid': self.device_uid}

class BlockDiagramNode:
    x: float
    y: float
    tpe: str
    device_uid: int
    subdiagram: "BlockDiagram | None" = None   # 👈 NEW

    def get_node_dict(self):
        data = {
            'x': self.x,
            'y': self.y,
            'tpe': self.tpe,         # fix: use 'tpe' not 'type'
            'device_uid': self.device_uid
        }
        if self.subdiagram:
            data['subdiagram'] = self.subdiagram.get_node_dict()
        return data


@dataclass
class BlockDiagramConnection:
    from_uid: int
    to_uid: int
    port_number_from: int
    port_number_to: int
    color: str

    def get_connection_dict(self):
        """
        get as a dictionary point
        :return:
        """
        return {'from_uid': self.from_uid,
                'to_uid': self.to_uid,
                'port_number_from': self.port_number_from,
                'port_number_to': self.port_number_to,
                'color': self.color}




class BlockDiagram:
    """
    Diagram
    """

    def __init__(self, idtag=None, name=''):
        """

        :param name: Diagram name
        """
        self.node_data: Dict[int, BlockDiagramNode] = dict()
        self.con_data: List[BlockDiagramConnection] = list()

    def add_node(self, x: float, y: float, device_uid: int, tpe: str):
        """

        :param x:
        :param y:
        :param device_uid:
        :param tpe:
        :return:
        """
        self.node_data[device_uid] = BlockDiagramNode(
            x=x,
            y=y,
            tpe=tpe,
            device_uid=device_uid
        )

    def add_branch(self, device_uid_from: int, device_uid_to: int,
                   port_number_from: int, port_number_to: int, color: str):
        """

        :param device_uid_from:
        :param device_uid_to:
        :param port_number_from:
        :param port_number_to:
        :param color:
        :return:
        """
        self.con_data.append(
            BlockDiagramConnection(
                from_uid=device_uid_from,
                to_uid=device_uid_to,
                port_number_from=port_number_from,
                port_number_to=port_number_to,
                color=color
            )
        )
    def get_node_data_dict(self) -> Dict[int, Dict[str, int]]:
        graph_info ={device_uid: node.get_node_dict() for device_uid, node in self.node_data.items()}
        return graph_info

    def get_con_data_dict(self) -> Dict[int, Dict[str, int]]:
        graph_info = {index: connection.get_connection_dict() for index, connection in enumerate(self.con_data)}
        return graph_info

    def parse_nodes(self, nodes_data) -> None:
        """
        Parse file data ito this class
        :param nodes_data: json dictionary
         """

        self.node_data = dict()
        for uid, node in nodes_data.items():
            self.node_data[uid] = BlockDiagramNode(
            x=node['x'],
            y=node['y'],
            tpe=node['tpe'],
            device_uid=node['device_uid']
        )

    def parse_branches(self, con_data)-> None:
        """
        Parse file data ito this class
        :param con_data: json dictionary
         """

        self.con_data = list()
        for idx, node in con_data.items():
            self.con_data.append(BlockDiagramConnection(
                from_uid=con_data['from_uid'],
                to_uid=con_data['to_uid'],
                port_number_from=con_data['port_number_from'],
                port_number_to=con_data['port_number_to'],
                color=con_data['color'],
            ))




class DynamicModelHost(EditableDevice):
    """
    This class serves to give flexible access to either a template or a custom model
    """

    def __init__(self, name=""):

        super().__init__(name=name,
                         idtag=None,
                         code="",
                         device_type=DeviceType.DynamicModelHostDevice)

        self._template: Block | None = None

        # a custom model always exits although it may be empty
        self._custom_model: Block = Block()
        self._diagram: BlockDiagram = BlockDiagram()

    @property
    def template(self):
        return self._template

    @template.setter
    def template(self, val: Block | RmsModelTemplate):

        if isinstance(val, RmsModelTemplate):
            self._template = val.block

        elif isinstance(val, Block):
            self._template = val

        else:
            raise ValueError(f"Cannot set template with {val}")

    @property
    def custom_model(self):
        return self._custom_model

    @property
    def model(self) -> Block:
        """
        Returns whatever is available with preference to the custom model if any
        :return: DynamicModel (even if it is empty)
        """
        if self.template is None:
            return self.custom_model
        else:
            return self.template

    @model.setter
    def model(self, val: Block | RmsModelTemplate):

        if isinstance(val, RmsModelTemplate):
            self.template = val.block

        elif isinstance(val, Block):
            self._custom_model = val

        else:
            raise ValueError(f"Cannot set model with {val}")

    @property
    def diagram(self) -> BlockDiagram:

        return self._diagram

    def to_dict(self) -> Dict[str, int | Dict[str, List[Dict[str, Any]]]]:
        """
        Generate a dictionary to save
        :return: Data to save
        """
        return {
            "template": self.template.uid if self.template is not None else None,
            "custom_model": self.custom_model.to_dict()
        }

    def parse(self, data: Dict[str, str | Dict[str, List[Dict[str, Any]]]],
              models_dict: Dict[str, RmsModelTemplate]):
        """
        Parse the data
        :param data: data generated by to_dict
        :param models_dict: dictionary of DynamicModel to find the template reference
        """
        template_id = data.get("template", None)
        if template_id is not None:
            self.template = models_dict.get(template_id, None)

        custom_data = data.get("custom_model", None)
        self._custom_model = Block.parse(data=custom_data)

    def empty(self):
        if self._template is None:
            return self._custom_model.empty()
        else:
            return self._template.empty()

    def __eq__(self, other):
        if isinstance(other, DynamicModelHost):

            if self.template is None:
                if other.template is None:
                    return self.custom_model == other.custom_model
                else:
                    return False
            else:
                if other.template is None:
                    return False
                else:
                    return self.template.uid == other.template.uid
        else:
            return False

    def copy(self) -> "DynamicModelHost":
        """
        Deep copy of DynamicModelHost
        :return: DynamicModelHost
        """
        obj = DynamicModelHost()
        obj._custom_model = self._custom_model.copy()
        obj._template = self._template
        return obj
