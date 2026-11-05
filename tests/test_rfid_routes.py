"""What the reader reports for the UI: no reader is not the same as no accessory."""

import asyncio
import threading
import webbrowser

import pytest

from reachy_mini_conversation_app import rfid_routes
from reachy_mini_conversation_app.personality_tag import to_tag_token
from reachy_mini_conversation_app.nfc_daemon_client import NfcTagSnapshot, NfcDaemonClient


@pytest.fixture
def controller(monkeypatch):
    """Build a controller whose move datasets and daemon calls never leave the machine."""
    monkeypatch.setattr(rfid_routes, "_load_move_dataset", lambda repo_id: None)
    return rfid_routes.RfidController(
        get_handler=lambda: pytest.fail("the handler must not be needed to report state"),
        get_loop=lambda: None,
        robot=None,
    )


def fake_reader(monkeypatch, *, connected, tag=None, error=None, error_code=None, enabled=None):
    """Answer the daemon's status and tag endpoints without a reader.

    ``enabled`` left as None leaves the field out, as daemons without the
    switch do.
    """
    switch = {} if enabled is None else {"enabled": enabled}
    monkeypatch.setattr(
        NfcDaemonClient,
        "get_status",
        lambda self: {
            **switch,
            "connected": connected,
            "chip_detected": connected,
            "driver_available": True,
            "port": "/dev/nfc",
            "error": error,
            "error_code": error_code,
        },
    )
    absent = NfcTagSnapshot(present=False, content=None, blank=False)
    monkeypatch.setattr(NfcDaemonClient, "get_tag", lambda self: tag or absent)


def test_an_unplugged_reader_is_unavailable_not_empty(controller, monkeypatch):
    """The regression: an unplugged module read as "None", as if only a tag were missing."""
    fake_reader(monkeypatch, connected=False)
    assert controller.snapshot()["accessory"]["state"] == "unavailable"
    assert controller.poll_once()["accessory"]["state"] == "unavailable"


def test_a_connected_reader_with_nothing_on_it_is_empty(controller, monkeypatch):
    """A reader that answers but holds no tag is the "None" the badge shows."""
    fake_reader(monkeypatch, connected=True)
    assert controller.snapshot()["accessory"]["state"] == "none"


@pytest.mark.parametrize(
    "tag, state",
    [
        (NfcTagSnapshot(present=True, content=None, blank=True), "blank"),
        (NfcTagSnapshot(present=True, content="written-elsewhere", blank=False), "unknown"),
        (NfcTagSnapshot(present=True, content=to_tag_token("default"), blank=False), "known"),
    ],
)
def test_a_tag_on_a_connected_reader_is_described_by_what_it_carries(controller, monkeypatch, tag, state):
    """Each tag state stays distinct from both "none" and "unavailable"."""
    fake_reader(monkeypatch, connected=True, tag=tag)
    assert controller.snapshot()["accessory"]["state"] == state


def test_the_daemon_reason_for_a_down_link_reaches_the_panel(controller, monkeypatch):
    """Not connected on its own leaves a user guessing; the daemon says which it is."""
    fake_reader(monkeypatch, connected=False, error="no NFC reader board found", error_code="NO_BOARD")
    assert controller.connection_status()["error"] == "no NFC reader board found"
    assert controller.last_status()["error"] == "no NFC reader board found"
    assert controller.last_status()["error_code"] == "NO_BOARD"


@pytest.mark.parametrize("enabled, expected", [(False, False), (True, True), (None, True)])
def test_a_reader_switched_off_is_told_apart_from_a_missing_one(controller, monkeypatch, enabled, expected):
    """Off, the daemon stops looking for the board: the panel must not offer to buy one."""
    fake_reader(monkeypatch, connected=False, enabled=enabled)
    snapshot = controller.snapshot()
    assert snapshot["enabled"] is expected
    assert snapshot["accessory"]["state"] == "unavailable"


@pytest.fixture
def add_on_store(controller):
    """Register the rfid.* methods and return the one that opens the add-on store."""
    methods = {}

    class FakeRpc:
        def register(self, name, handler):
            methods[name] = handler

    rfid_routes.register_rfid_methods(FakeRpc(), controller)
    return methods["rfid.open_add_on_store"]


@pytest.mark.asyncio
async def test_the_store_opens_on_the_machine_running_the_app(monkeypatch, add_on_store):
    """The panel cannot raise a browser window from its webview, so the app does it."""
    asked = []
    monkeypatch.setattr(webbrowser, "open", lambda url: bool(asked.append(url)) or True)
    result = await add_on_store({})
    assert asked == [rfid_routes.ADD_ON_STORE_URL]
    assert result == {"opened": True, "url": rfid_routes.ADD_ON_STORE_URL}


@pytest.mark.asyncio
async def test_a_host_with_no_browser_still_hands_back_the_address(monkeypatch, add_on_store):
    """A robot with no screen of its own: the panel shows the URL instead."""
    monkeypatch.setattr(webbrowser, "open", lambda url: False)
    assert await add_on_store({}) == {"opened": False, "url": rfid_routes.ADD_ON_STORE_URL}


def test_an_accessory_left_on_the_head_names_the_personality_to_start_as(monkeypatch):
    """The app reads the reader before building its handler, so it starts as that one."""
    fake_reader(
        monkeypatch,
        connected=True,
        tag=NfcTagSnapshot(present=True, content=to_tag_token("default"), blank=False),
    )
    assert rfid_routes.accessory_personality_on_reader() == "default"


@pytest.mark.parametrize(
    "tag",
    [
        None,
        NfcTagSnapshot(present=True, content=None, blank=True),
        NfcTagSnapshot(present=True, content="written-elsewhere", blank=False),
        NfcTagSnapshot(present=True, content=to_tag_token("gone_from_this_robot"), blank=False),
    ],
    ids=["no accessory", "blank", "written elsewhere", "personality this robot lacks"],
)
def test_nothing_to_start_from_leaves_the_saved_personality_alone(monkeypatch, tag):
    """Anything the robot cannot act on reads the same way: the app starts as usual."""
    fake_reader(monkeypatch, connected=True, tag=tag)
    assert rfid_routes.accessory_personality_on_reader() is None


def test_a_robot_without_the_nfc_driver_is_not_asked_for_a_tag(monkeypatch):
    """No driver means no reader on this robot, so the launch never waits on one."""
    monkeypatch.setattr(NfcDaemonClient, "get_status", lambda self: {"driver_available": False})
    monkeypatch.setattr(NfcDaemonClient, "get_tag", lambda self: pytest.fail("the tag must not be read"))
    assert rfid_routes.accessory_personality_on_reader() is None


class _StubHandler:
    """Just enough handler for the tag paths: switches run on the test loop, taking ``delay``."""

    class _Deps:
        blank_tag_present = False
        pending_nfc_write = None
        recently_written_codes: set = set()

    def __init__(self, delay=0.0):
        self.deps = self._Deps()
        self.deps.recently_written_codes = set()
        self.applied = []
        self._delay = delay

    def abort_nfc_collection(self):
        return None

    async def apply_personality(self, profile, on_applied=None):
        # Like the real handler: in effect at once, then a slow session restart.
        self.applied.append(profile)
        if on_applied is not None:
            on_applied()
        await asyncio.sleep(self._delay)
        return "ok"


@pytest.fixture
def conversation_loop():
    """Run an event loop on its own thread, standing in for the conversation loop."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    loop.close()


def removal_controller(monkeypatch, conversation_loop, *, initial_personality=None, default_personality=None):
    """Build a controller whose switches run on a real loop, without a handler.

    Returns the controller and the list of personalities it announced.
    """
    monkeypatch.setattr(rfid_routes, "_load_move_dataset", lambda repo_id: None)
    announced = []
    controller = rfid_routes.RfidController(
        get_handler=lambda: pytest.fail("the handler is passed in, not fetched"),
        get_loop=lambda: conversation_loop,
        robot=None,
        on_personality_applied=announced.append,
        get_default_personality=lambda: default_personality,
        initial_personality=initial_personality,
    )
    monkeypatch.setattr(controller, "_queue_move", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_run_on_loop", lambda coro, description, **kw: True)
    return controller, announced


def test_taking_off_the_accessory_the_app_started_with_reverts_to_the_default(monkeypatch, conversation_loop):
    """The controller applied nothing itself, so it has to be told what it started as."""
    controller, announced = removal_controller(monkeypatch, conversation_loop, initial_personality="pirate")

    controller._on_tag_removed(_StubHandler())

    assert announced == [None]


def test_taking_off_an_accessory_goes_back_to_the_personality_set_as_default(monkeypatch, conversation_loop):
    """The fallback is whatever this instance was told to start as, not the built-in one."""
    controller, announced = removal_controller(
        monkeypatch, conversation_loop, initial_personality="pirate", default_personality="king_reachy_maximus"
    )
    monkeypatch.setattr(rfid_routes, "list_personalities", lambda: ["default", "king_reachy_maximus", "pirate"])

    controller._on_tag_removed(_StubHandler())

    assert announced == ["king_reachy_maximus"]


def test_a_default_personality_since_deleted_falls_back_to_the_built_in_one(monkeypatch, conversation_loop):
    """Reverting to a personality this robot no longer has would fail and strand the accessory's."""
    controller, announced = removal_controller(
        monkeypatch, conversation_loop, initial_personality="pirate", default_personality="deleted_since"
    )
    monkeypatch.setattr(rfid_routes, "list_personalities", lambda: ["default", "pirate"])

    controller._on_tag_removed(_StubHandler())

    assert announced == [None]


def test_an_accessory_carrying_the_default_personality_changes_nothing_when_removed(monkeypatch, conversation_loop):
    """Restarting the backend to arrive at what is already running only costs a silence."""
    controller, announced = removal_controller(
        monkeypatch,
        conversation_loop,
        initial_personality="king_reachy_maximus",
        default_personality="king_reachy_maximus",
    )
    monkeypatch.setattr(rfid_routes, "list_personalities", lambda: ["default", "king_reachy_maximus"])

    controller._on_tag_removed(_StubHandler())

    assert announced == []


def test_a_controller_that_started_at_the_default_has_nothing_to_revert(monkeypatch, conversation_loop):
    """No accessory at launch: taking nothing off must not restart the backend."""
    controller, announced = removal_controller(monkeypatch, conversation_loop)

    controller._on_tag_removed(_StubHandler())

    assert announced == []


def test_a_switch_is_announced_before_the_session_restart_ends(monkeypatch, conversation_loop):
    """Restarting the voice session takes seconds: clients hear of the switch when it is in effect."""
    monkeypatch.setattr(rfid_routes, "list_personalities", lambda: ["default", "pirate"])
    controller, announced = removal_controller(monkeypatch, conversation_loop)
    # Longer than the poll waits: the call returns while the restart still runs.
    monkeypatch.setattr(rfid_routes, "PERSONALITY_APPLY_WAIT_S", 0.05)

    controller._on_tag_read(_StubHandler(delay=0.3), to_tag_token("pirate"))

    assert announced == ["pirate"]
    # Taking the accessory off meanwhile still reverts it: the switch was recorded.
    assert controller._current_personality == "pirate"
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0.5), conversation_loop).result(timeout=2)
    assert announced == ["pirate"]


def test_an_accessory_taken_off_mid_switch_has_the_last_word(monkeypatch, conversation_loop):
    """The revert is announced after the switch it interrupted, and nothing comes after it."""
    monkeypatch.setattr(rfid_routes, "PERSONALITY_APPLY_WAIT_S", 0.05)
    monkeypatch.setattr(rfid_routes, "list_personalities", lambda: ["default", "pirate"])
    controller, announced = removal_controller(monkeypatch, conversation_loop)
    handler = _StubHandler(delay=0.3)

    controller._on_tag_read(handler, to_tag_token("pirate"))
    controller._on_tag_removed(handler)

    asyncio.run_coroutine_threadsafe(asyncio.sleep(0.8), conversation_loop).result(timeout=2)
    assert announced == ["pirate", None]
    assert controller._current_personality is None


def test_a_personality_that_cannot_be_loaded_is_not_announced(monkeypatch, conversation_loop):
    """The handler then keeps the previous one: telling clients otherwise would be a lie."""
    monkeypatch.setattr(rfid_routes, "list_personalities", lambda: ["default", "pirate"])
    controller, announced = removal_controller(monkeypatch, conversation_loop)

    class Unloadable(_StubHandler):
        async def apply_personality(self, profile, on_applied=None):
            return "Failed to apply personality: unreadable"

    controller._on_tag_read(Unloadable(), to_tag_token("pirate"))

    assert announced == []
    assert controller._current_personality is None


def _tag(content=None, *, blank=False):
    return NfcTagSnapshot(present=True, content=content, blank=blank)


ABSENT = NfcTagSnapshot(present=False, content=None, blank=False)


@pytest.fixture
def events(controller, monkeypatch):
    """Record which handler each reader change reaches, in order."""
    seen = []
    monkeypatch.setattr(controller, "_get_handler", lambda: None)
    monkeypatch.setattr(controller, "_on_tag_removed", lambda handler: seen.append(("removed",)))
    monkeypatch.setattr(controller, "_on_tag_read", lambda handler, code: seen.append(("read", code)))
    monkeypatch.setattr(
        controller, "_on_write_result", lambda handler, success, code: seen.append(("write", success, code))
    )
    return seen


@pytest.mark.parametrize(
    "previous, tag, expected",
    [
        (None, _tag("hf_pirate"), []),
        (ABSENT, ABSENT, []),
        (_tag("hf_pirate"), ABSENT, [("removed",)]),
        (ABSENT, _tag("hf_pirate"), [("read", "hf_pirate")]),
        (ABSENT, _tag(blank=True), [("read", "")]),
        (_tag("hf_pirate"), _tag(blank=True), [("read", "")]),
        (_tag(blank=True), _tag("hf_pirate"), [("read", "hf_pirate")]),
        (_tag("hf_pirate"), _tag("hf_chef"), [("read", "hf_chef")]),
        (_tag("hf_pirate"), _tag("hf_pirate"), []),
        (_tag(blank=True), _tag(blank=True), []),
    ],
    ids=[
        "first read",
        "still nothing",
        "removed",
        "placed",
        "blank placed",
        "erased in place",
        "written in place",
        "changed in place",
        "unchanged",
        "still blank",
    ],
)
def test_each_reader_change_reaches_its_handler(controller, events, previous, tag, expected):
    """Two consecutive reads map to at most one tag event."""
    controller._process([], previous, tag)
    assert events == expected


def test_write_results_are_handled_before_the_tag_change(controller, events):
    """A write and the read of the tag it produced can land in the same poll.

    The write goes first: it arms the speech gate that the delayed personality
    switch, started by the read, then waits on.
    """
    controller._process([(True, "WRITE_OK")], _tag(blank=True), _tag("usr_pirate"))
    assert events == [("write", True, "WRITE_OK"), ("read", "usr_pirate")]


def _unreadable(uid="04A1"):
    return NfcTagSnapshot(present=True, content=None, blank=False, readable=False, uid=uid)


def _read_in_turn(monkeypatch, controller, readings):
    """Poll the controller once per reading, the daemon answering each in turn."""
    pending = list(readings)
    fake_reader(monkeypatch, connected=True)
    # On this controller's client only: a poll thread another test left running
    # would otherwise eat the readings.
    monkeypatch.setattr(controller.client, "get_tag", lambda: pending.pop(0))
    return [controller.poll_once()["accessory"]["state"] for _ in readings]


def test_an_accessory_that_fails_a_read_stays_linked(controller, events, monkeypatch):
    """A read that fails under a linked accessory is not a blank one: nothing happens."""
    pirate = NfcTagSnapshot(present=True, content="hf_pirate", blank=False, uid="04A1")
    states = _read_in_turn(monkeypatch, controller, [ABSENT, pirate, _unreadable(), _unreadable(), pirate])
    assert events == [("read", "hf_pirate")]
    assert "blank" not in states


def test_an_accessory_unreadable_on_arrival_waits_for_its_content(controller, events, monkeypatch):
    """Arriving unreadable, the accessory is acted on once its content comes through, not before."""
    pirate = NfcTagSnapshot(present=True, content="hf_pirate", blank=False, uid="04A1")
    states = _read_in_turn(monkeypatch, controller, [ABSENT, _unreadable(), pirate])
    assert events == [("read", "hf_pirate")]
    assert states[1] == "none"


def test_an_unreadable_tag_never_reaches_the_blank_tag_handling(controller, events, monkeypatch):
    """Taken as blank, it would start the personality conversation, or receive a pending code."""
    _read_in_turn(monkeypatch, controller, [ABSENT, _unreadable(), _unreadable()])
    assert events == []


def test_a_different_accessory_unreadable_in_place_of_the_last_is_a_removal(controller, events, monkeypatch):
    """The accessory that was linked is gone: the one replacing it is acted on once read."""
    pirate = NfcTagSnapshot(present=True, content="hf_pirate", blank=False, uid="04A1")
    _read_in_turn(monkeypatch, controller, [ABSENT, pirate, _unreadable(uid="0499")])
    assert events == [("read", "hf_pirate"), ("removed",)]


def test_a_truly_blank_accessory_is_still_blank(controller, events, monkeypatch):
    """A tag read as blank is still handled as one."""
    blank = NfcTagSnapshot(present=True, content=None, blank=True, uid="04A1")
    states = _read_in_turn(monkeypatch, controller, [ABSENT, blank])
    assert events == [("read", "")]
    assert states[-1] == "blank"


def test_the_client_tells_an_unreadable_tag_apart(monkeypatch):
    """The daemon's ``readable`` and ``uid`` reach the app; older daemons fall back on ``error``."""

    class Answer:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    client = NfcDaemonClient()
    body = {"present": True, "uid": "04A1", "blank": False, "readable": False, "error": "cannot read"}
    monkeypatch.setattr(client._session, "get", lambda *a, **k: Answer(body))
    tag = client.get_tag()
    assert (tag.readable, tag.uid) == (False, "04A1")

    body = {"present": True, "blank": False, "error": "cannot read"}
    assert not client.get_tag().readable
    body = {"present": True, "content": "hf_pirate"}
    assert client.get_tag().readable


class _FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


@pytest.mark.parametrize(
    "response, expected",
    [
        (_FakeResponse(503, {"detail": "NFC reader disabled"}), (False, "DISABLED")),
        (_FakeResponse(200, {"success": False, "error": "UNCERTAIN"}), (False, "UNCERTAIN")),
        (_FakeResponse(200, {"success": True}), (True, "WRITE_OK")),
    ],
)
def test_a_write_outcome_is_a_code_whatever_the_wording(monkeypatch, response, expected):
    """A switched-off reader is a 503: told by its status code, never by its message."""
    monkeypatch.setattr("requests.post", lambda *a, **kw: response)
    assert NfcDaemonClient().write_tag("abc") == expected


@pytest.mark.parametrize("code", ["DISABLED", "UNCERTAIN", "BUSY"])
def test_every_daemon_code_has_a_message(code):
    """Codes the daemon added for the NFC API reach a user as a sentence."""
    from reachy_mini_conversation_app.nfc_daemon_client import describe_write_error

    assert describe_write_error(code) != code
