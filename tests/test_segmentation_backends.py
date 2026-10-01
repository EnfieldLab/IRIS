import pytest
import numpy as np
from iris_viewer.engine.segmentation.cellpose_backend import CellposeBackend
from iris_viewer.engine.segmentation.stardist_backend import StardistBackend
from iris_viewer.engine.segmentation.instanseg_backend import InstanSegBackend
from iris_viewer.engine.segmentation.cellsam_backend import CellSAMBackend

@pytest.fixture
def dummy_image():
    # Create a small dummy image: 3 channels, 128x128
    # Using uint8 for broad compatibility
    img = np.random.randint(0, 255, (3, 128, 128), dtype=np.uint8)
    return img

def test_cellpose_backend(dummy_image):
    backend = CellposeBackend()
    if not backend.is_available():
        pytest.skip("Cellpose not installed")
        
    res = backend.segment(
        image=dummy_image,
        nuclear_channel=0,
        membrane_channel=1,
        diameter=15.0,
        device='cpu',
        model_type='cyto3'
    )
    assert 'nuclear' in res
    assert res['nuclear'].shape == (128, 128)
    
def test_stardist_backend(dummy_image):
    backend = StardistBackend()
    if not backend.is_available():
        pytest.skip("StarDist not installed")
        
    res = backend.segment(
        image=dummy_image,
        nuclear_channel=0,
        membrane_channel=None,
        diameter=None,
        device='cpu',
        model_name='2D_versatile_fluo'
    )
    assert 'nuclear' in res
    assert res['nuclear'].shape == (128, 128)
    
def test_instanseg_backend(dummy_image):
    backend = InstanSegBackend()
    if not backend.is_available():
        pytest.skip("InstanSeg not installed")
        
    res = backend.segment(
        image=dummy_image,
        nuclear_channel=0,
        membrane_channel=1,
        diameter=None,
        device='cpu',
        model_name='fluorescence_nuclei_and_cells'
    )
    assert 'nuclear' in res
    assert res['nuclear'].shape == (128, 128)

def test_cellsam_backend(dummy_image):
    backend = CellSAMBackend()
    if not backend.is_available():
        pytest.skip("CellSAM not installed")
        
    try:
        res = backend.segment(
            image=dummy_image,
            nuclear_channel=0,
            membrane_channel=None,
            diameter=None,
            device='cpu'
        )
        assert 'nuclear' in res
        assert res['nuclear'].shape == (128, 128)
    except ValueError as e:
        if "DEEPCELL_ACCESS_TOKEN" in str(e):
            pytest.skip("Skipped CellSAM test: DEEPCELL_ACCESS_TOKEN not set.")
        else:
            raise
