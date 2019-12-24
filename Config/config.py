"""Utility functions and other shared chemgraph code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from tensorflow import gfile


def read_hparams(filename, defaults):
  """Reads HParams from JSON.
  Args:
    filename: String filename.
    defaults: HParams containing default values.
  Returns:
    HParams.
  Raises:
    gfile.Error: If the file cannot be read.
    ValueError: If the JSON record cannot be parsed.
  """
  with gfile.Open(filename) as f:
    logging.info('Reading HParams from %s', filename)
    return defaults.parse_json(f.read())


def write_hparams(hparams, filename):
    """Writes HParams to disk as JSON.
       Args:
       hparams: HParams.
       filename: String output filename.
    """
    with gfile.Open(filename, 'w') as f:
        f.write(hparams.to_json(indent=2, sort_keys=True, separators=(',', ': ')))