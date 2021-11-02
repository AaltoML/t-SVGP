import glob
import os
import sys
import traceback

import jupytext
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError


def _nbpath():
    this_dir = os.path.dirname(__file__)
    return os.path.join(this_dir, "../docs/notebooks/")


def test_notebook_dir_exists():
    assert os.path.isdir(_nbpath())


def get_notebooks():
    """
    Returns all notebooks in `_nbpath`
    """
    all_notebooks = glob.iglob(os.path.join(_nbpath(), "**", "*.py"), recursive=True)
    notebooks_to_test = [nb for nb in all_notebooks]
    return notebooks_to_test

def _preproc():
    pythonkernel = "python" + str(sys.version_info[0])
    return ExecutePreprocessor(timeout=300, kernel_name=pythonkernel, interrupt_on_timeout=True)


def _exec_notebook(notebook_filename):
    with open(notebook_filename) as notebook_file:
        nb = jupytext.read(notebook_file, as_version=nbformat.current_nbformat)
        try:
            meta_data = {"path": os.path.dirname(notebook_filename)}
            _preproc().preprocess(nb, {"metadata": meta_data})
        except CellExecutionError as cell_error:
            traceback.print_exc(file=sys.stdout)
            msg = "Error executing the notebook {0}. See above for error.\nCell error: {1}"
            pytest.fail(msg.format(notebook_filename, str(cell_error)))


@pytest.mark.notebooks
@pytest.mark.parametrize("notebook_file", get_notebooks())
def test_notebook(notebook_file):
    _exec_notebook(notebook_file)


def test_has_notebooks():
    assert len(get_notebooks()) >= 2, "there are probably some notebooks that were not discovered"