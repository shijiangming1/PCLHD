from __future__ import absolute_import
import warnings

from .market1501 import Market1501
from .msmt17 import MSMT17
from .personx import PersonX
from .veri import VeRi
from .dukemtmcreid import DukeMTMCreID
from .sysu_all import sysu_all
from .sysu_ir import sysu_ir
from .sysu_rgb import sysu_rgb
from .regdb_ir import regdb_ir
from .regdb_rgb import regdb_rgb
__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'personx': PersonX,
    'veri': VeRi,
    'dukemtmcreid': DukeMTMCreID,
    'sysu_all': sysu_all,
    'sysu_ir':sysu_ir,
    'sysu_rgb':sysu_rgb,
    'regdb_ir':regdb_ir,
    'regdb_rgb':regdb_rgb
}


def names():
    return sorted(__factory.keys())


def create(name, root,trial=0, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. 
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, trial=trial, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
