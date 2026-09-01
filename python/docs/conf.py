# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import enum
import inspect
import os
import sys

from sphinx.ext.autodoc import AttributeDocumenter, ClassDocumenter
from sphinx.util.docstrings import prepare_docstring

sys.path.insert(0, os.path.abspath('../../..'))

project = 'PyCuVSLAM'
copyright = '2026, NVIDIA CORPORATION'
author = 'NVIDIA CORPORATION'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
# Style from Isaac ROS
html_css_files = [
    'css/custom.css',
]

autodoc_member_order = 'bysource'
autodoc_typehints = 'both'
autodoc_typehints_format = 'short'
autodoc_typehints_description_target = 'all'
autodoc_class_signature = 'mixed'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None


class NanobindEnumValueDocumenter(AttributeDocumenter):
    """Document per-value enum docstrings exposed by nanobind."""

    objtype = 'nanobind-enum-value'
    directivetype = 'attribute'
    priority = AttributeDocumenter.priority + 1

    @classmethod
    def can_document_member(_cls, member, _membername, _isattr, _parent) -> bool:
        return isinstance(member, enum.Enum)

    def get_doc(self):
        enum_value = self.get_attr(self.parent, self.object_name)
        docstring = getattr(enum_value, '__doc__', None)
        if not docstring:
            return []
        tab_width = self.directive.state.document.settings.tab_width
        return [prepare_docstring(inspect.cleandoc(docstring), tab_width)]


class NanobindClassDocumenter(ClassDocumenter):
    """Preserve binding definition order for extension classes."""

    def sort_members(self, documenters, order):
        if self.object.__module__ == 'cuvslam.pycuvslam':
            definition_order = {name: index for index, name in enumerate(self.object.__dict__)}

            def member_position(entry) -> int:
                name = entry[0].name.rsplit('.', 1)[-1]
                return definition_order.get(name, len(definition_order))

            documenters.sort(
                key=member_position)
            return documenters
        return super().sort_members(documenters, order)


def setup(app):
    app.add_autodocumenter(NanobindEnumValueDocumenter)
    app.add_autodocumenter(NanobindClassDocumenter, override=True)
