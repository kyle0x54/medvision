import os
import pytest
import medvision as mv


def gen_path(*paths):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    return os.path.join(data_dir, *paths)


@pytest.mark.parametrize('given, expected', [
    (gen_path('dicoms', 'brain_001.dcm'), True),
    (gen_path('pngs', 'Blue-Ogi.png'), False),
    (gen_path('dicoms'), False),
    (gen_path('texts', 'null.txt'), False)
])
def test_isdicom(given, expected):
    assert mv.isdicom(given) == expected


@pytest.mark.parametrize('given, expected', [
    (gen_path('dicoms'), True),
    (gen_path('dicoms', 'brain_001.dcm'), False),
    (gen_path('pngs'), False),
    (gen_path('medvision'), False)
])
def test_isdicomdir(given, expected):
    assert mv.isdicomdir(given) == expected
