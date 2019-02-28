from collections import OrderedDict
from natsort import natsorted


def make_dsmd(data):
    """ Make a dataset metadata.

    Dataset Metadata is a key-value pairs describing a dataset.
    For example, a dataset metadata can be a dictionary looks like
    {
        'data/1.png': 1,
        'data/2.png': 0,
        ...
    }

    A dataset metadata file defines the mapping from files
    to annotations. Several types of data metadata files are listed below.

    +-------------------------------------------------------+
    | A. Single Label Classification Dataset Metadata File  |
    +-------------------------------------------------------+
    |data/1.png,1                                           |
    |data/2.png,0                                           |
    |...                                                    |
    +-------------------------------------------------------+

    +-------------------------------------------------------+
    | B. Multi-Label Classification Dataset Metadata File   |
    +-------------------------------------------------------+
    |data/1.png,1,0,1                                       |
    |data/2.png,0,1,0                                       |
    |...                                                    |
    +-------------------------------------------------------+

    +-------------------------------------------------------+
    | C. Segmentation Dataset Metadata File                 |
    +-------------------------------------------------------+
    |data/1.png,data/1_mask.png                             |
    |data/2.png,data/2_mask.png                             |
    |...                                                    |
    +-------------------------------------------------------+

    +-------------------------------------------------------+
    | D. Detection Dataset Metadata File A (bbox only)      |
    +-------------------------------------------------------+
    |data/1.png,170,146,397,681,cat                         |
    |data/1.png,473,209,723,673,cat                         |
    |data/1.png,552,167,745,272,dog                         |
    |data/2.png,578,267,624,304,dog                         |
    |data/3.png,,,,,                                        |
    |...                                                    |
    +-------------------------------------------------------+

    +-------------------------------------------------------+
    | E. Detection Dataset Metadata File B (bbox with score)|
    +-------------------------------------------------------+
    |data/1.png,170,146,397,681,0.5,cat                     |
    |data/1.png,473,209,723,673,0.2,cat                     |
    |data/1.png,552,167,745,272,0.7,dog                     |
    |data/2.png,578,267,624,304,0.9,dog                     |
    |data/3.png,,,,,                                        |
    |...                                                    |
    +-------------------------------------------------------+

    Args:
        data (dict): dataset metadata.
    """
    if isinstance(data, dict):
        dsmd = OrderedDict(natsorted(data.items()))
        return dsmd
    else:
        raise ValueError('dsmd only support dict type')
