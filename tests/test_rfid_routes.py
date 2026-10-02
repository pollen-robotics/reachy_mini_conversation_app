"""What the reader reports for the UI: no reader is not the same as no accessory."""

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


def fake_reader(monkeypatch, *, connected, tag=None, error=None, enabled=None):
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
    fake_reader(monkeypatch, connected=False, error="no NFC reader board found")
    assert controller.connection_status()["error"] == "no NFC reader board found"
    assert controller.last_status()["error"] == "no NFC reader board found"


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
    """Just enough handler for the tag-removal path, which never reaches the loop."""

    class _Deps:
        blank_tag_present = False
        pending_nfc_write = None

    def __init__(self):
        self.deps = self._Deps()

    def abort_nfc_collection(self):
        return None

    def apply_personality(self, profile):
        return None


def removal_controller(monkeypatch, *, initial_personality=None, default_personality=None):
    """Build a controller whose revert is observable without a handler or a loop.

    Returns the controller and the list of personalities it announced.
    """
    monkeypatch.setattr(rfid_routes, "_load_move_dataset", lambda repo_id: None)
    announced = []
    controller = rfid_routes.RfidController(
        get_handler=lambda: pytest.fail("the handler is passed in, not fetched"),
        get_loop=lambda: None,
        robot=None,
        on_personality_applied=announced.append,
        get_default_personality=lambda: default_personality,
        initial_personality=initial_personality,
    )
    monkeypatch.setattr(controller, "_queue_move", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "_run_on_loop", lambda coro, description, **kw: True)
    return controller, announced


def test_taking_off_the_accessory_the_app_started_with_reverts_to_the_default(monkeypatch):
    """The controller applied nothing itself, so it has to be told what it started as."""
    controller, announced = removal_controller(monkeypatch, initial_personality="pirate")

    controller._on_tag_removed(_StubHandler())

    assert announced == [None]


def test_taking_off_an_accessory_goes_back_to_the_personality_set_as_default(monkeypatch):
    """The fallback is whatever this instance was told to start as, not the built-in one."""
    controller, announced = removal_controller(
        monkeypatch, initial_personality="pirate", default_personality="king_reachy_maximus"
    )
    monkeypatch.setattr(rfid_routes, "list_personalities", lambda: ["default", "king_reachy_maximus", "pirate"])

    controller._on_tag_removed(_StubHandler())

    assert announced == ["king_reachy_maximus"]


def test_a_default_personality_since_deleted_falls_back_to_the_built_in_one(monkeypatch):
    """Reverting to a personality this robot no longer has would fail and strand the accessory's."""
    controller, announced = removal_controller(
        monkeypatch, initial_personality="pirate", default_personality="deleted_since"
    )
    monkeypatch.setattr(rfid_routes, "list_personalities", lambda: ["default", "pirate"])

    controller._on_tag_removed(_StubHandler())

    assert announced == [None]


def test_an_accessory_carrying_the_default_personality_changes_nothing_when_removed(monkeypatch):
    """Restarting the backend to arrive at what is already running only costs a silence."""
    controller, announced = removal_controller(
        monkeypatch, initial_personality="king_reachy_maximus", default_personality="king_reachy_maximus"
    )
    monkeypatch.setattr(rfid_routes, "list_personalities", lambda: ["default", "king_reachy_maximus"])

    controller._on_tag_removed(_StubHandler())

    assert announced == []


def test_a_controller_that_started_at_the_default_has_nothing_to_revert(monkeypatch):
    """No accessory at launch: taking nothing off must not restart the backend."""
    controller, announced = removal_controller(monkeypatch)

    controller._on_tag_removed(_StubHandler())

    assert announced == []
