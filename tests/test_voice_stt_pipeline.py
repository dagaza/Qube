from core.voice_stt_pipeline import bump_voice_stt_epoch, is_voice_stt_job_current


def test_bump_voice_stt_epoch_increments():
    assert bump_voice_stt_epoch(0) == 1
    assert bump_voice_stt_epoch(3) == 4


def test_is_voice_stt_job_current_accepts_matching_epoch():
    assert is_voice_stt_job_current(2, 2) is True


def test_is_voice_stt_job_current_rejects_stale_epoch():
    assert is_voice_stt_job_current(1, 2) is False


def test_is_voice_stt_job_current_rejects_zero_epoch():
    assert is_voice_stt_job_current(0, 0) is False
    assert is_voice_stt_job_current(0, 1) is False
