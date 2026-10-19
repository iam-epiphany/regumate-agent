from pathlib import Path

import pytest

from dotenv import load_dotenv


# 让 pytest 与运行时行为一致：项目根目录 .env 注入环境变量（不覆盖已存在
# 的显式环境变量）。缺失 .env 时静默跳过，测试不受影响。
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)


@pytest.fixture(autouse=True)
def isolate_qa_worker(monkeypatch, request):
    """Task tests execute workers explicitly; do not leak a daemon into other tests."""
    from backend.app.services import qa_task_service
    import backend.main as main
    if 'live_qa_worker' not in request.fixturenames:
        monkeypatch.setattr(qa_task_service, 'start_qa_task_worker', lambda: None)
    monkeypatch.setattr(main, 'start_qa_task_worker', lambda: None)


@pytest.fixture
def live_qa_worker(monkeypatch):
    from queue import Queue, Empty
    from threading import Event, enumerate as threads
    from backend.app.services import qa_task_service as tasks
    stop = Event()
    before = set(threads())
    monkeypatch.setattr(tasks, '_QUEUE', Queue(maxsize=16))
    monkeypatch.setattr(tasks, '_PENDING', set())
    monkeypatch.setattr(tasks, '_STARTED', False)
    def bounded_worker():
        while not stop.is_set():
            try:
                task_id = tasks._QUEUE.get(timeout=0.05)
            except Empty:
                continue
            try:
                tasks._run_task(task_id)
            finally:
                with tasks._LOCK:
                    tasks._PENDING.discard(task_id)
                tasks._QUEUE.task_done()
    monkeypatch.setattr(tasks, '_worker_loop', bounded_worker)
    yield
    stop.set()
    for thread in set(threads())-before:
        if thread.name == 'regumate-qa-worker':
            thread.join(timeout=5)
            assert not thread.is_alive(), 'QA worker leaked beyond its test'


@pytest.fixture(autouse=True)
def isolate_document_cache():
    from backend.app.services.rag_service import clear_document_snapshot_cache
    clear_document_snapshot_cache()
    yield
    clear_document_snapshot_cache()


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    """Tests replace model boundaries explicitly; never inherit live API access."""
    import urllib.request
    def blocked(*args, **kwargs):
        raise OSError("Unit tests cannot make network requests")
    monkeypatch.setattr(urllib.request, "urlopen", blocked)


@pytest.fixture(autouse=True)
def offline_task_verifier(monkeypatch):
    """Protocol fixture; negative verifier cases replace it explicitly."""
    import json
    from backend.app.services import answer_validation_service
    def supported(_config, messages, **kwargs):
        payload = json.loads(messages[-1]['content'])
        return json.dumps({'complete': True, 'supported_claims': list(range(1, len(payload['claims'])+1)),
                           'missing_requirements': [], 'reasons': []})
    monkeypatch.setattr(answer_validation_service, 'chat_completion_content', supported)
