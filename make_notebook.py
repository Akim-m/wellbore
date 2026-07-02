"""Wrap kaggle_notebook.py into a Kaggle-ready .ipynb (one markdown + one code cell)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "kaggle_notebook.py"), encoding="utf-8").read()

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# ROGII Wellbore Geology Prediction\n",
            "Run All, then Save Version and Submit. Writes `submission.csv`.\n",
        ]},
        {"cell_type": "code", "metadata": {}, "execution_count": None,
         "outputs": [], "source": src.splitlines(keepends=True)},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = os.path.join(HERE, "kaggle_notebook.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
print(f"wrote {out}")
