"""
Nek5000 File Readers
====================

Module for reading Nek5000 simulation output files including:
- Field files (.fld, f00001)
- Mesh files
- Statistics files
"""

from .field_reader import Nek5000Reader, read_field_file, read_mesh, FieldData

__all__ = [
    'Nek5000Reader',
    'read_field_file',
    'read_mesh',
    'FieldData',
]
