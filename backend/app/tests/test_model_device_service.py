from backend.app.services.model_device_service import get_model_device_info


def test_model_device_info_reports_selected_device() -> None:
    info = get_model_device_info()

    assert info.selected_device in {"cpu", "cuda"}
    assert info.requested_device in {"auto", "cpu", "cuda"}
    assert isinstance(info.cuda_available, bool)
    if info.selected_device == "cuda":
        assert info.cuda_available is True
        assert info.cuda_device_count >= 1
