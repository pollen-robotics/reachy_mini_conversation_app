# External Profiles

Robot Comic supports loading persona profiles from a directory outside the repository.
This is the recommended way to use operator-specific personas — including any personas
that reference real individuals — while keeping the public codebase neutral.

## Why external profiles?

The profiles shipped in this repository (`profiles/`) are public and committed to the
open-source repo. Personas that draw on real comedians' performance styles, voice
references, or biographical material raise right-of-publicity considerations that make
them unsuitable for public distribution.

The operator's private persona library (kept in a separate private repository, not
distributed here) contains those profiles and is loaded at runtime via the environment
variable described below.

## `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY`

Set this environment variable to the absolute or relative path of a directory that
contains one subdirectory per persona:

```env
REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=/home/user/my_persona_library
```

At startup, the app merges the external directory with the built-in `profiles/`
directory. External profiles take precedence when a name is found in both places —
but built-in profile names are reserved, so the app will raise a `RuntimeError` if
an external profile attempts to shadow a built-in name (e.g. `default`).

### On a workstation

Add the variable to your `.env` file at the project root, or export it from your
shell profile before running the app:

```bash
export REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY=/path/to/private/persona/library
python -m robot_comic.main --sim
```

### On the robot

Add the variable to the systemd service environment or to the `.env` file that the
service loads. The path should be absolute so it resolves correctly regardless of
working directory.

## `LOCKED_PROFILE`

To lock the app to a single persona and disable all runtime profile switching, set
`LOCKED_PROFILE` in `src/robot_comic/config.py`:

```python
LOCKED_PROFILE: str | None = "house_comedian"
```

When set, the UI shows "(locked)" and disables the personality picker.
If the locked profile lives in the external directory, `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY`
must be set so the app can find it.

**Breaking change warning:** If you have deployed the app with `LOCKED_PROFILE` pointing
to a persona that has since been removed from the built-in profiles (e.g. one of the
named-comedian profiles that was deleted in this cleanup), the app will fail to start
until you update `LOCKED_PROFILE` to point at a profile that exists, either built-in or
in your external directory.

## External profile directory layout

An external profile directory should mirror the layout of built-in profiles in this
repository. Each profile is a subdirectory named after the profile identifier:

```text
my_persona_library/
├── persona_a/
│   ├── instructions.txt      # required — system prompt text
│   ├── tools.txt             # optional — one tool name per line
│   ├── voice.txt             # optional — voice ID or name
│   ├── gemini_live.txt       # optional — delivery guidance for Gemini Live
│   ├── gemini_tts.txt        # optional — TTS pacing guidance
│   ├── elevenlabs.txt        # optional — ElevenLabs voice config
│   ├── chatterbox.txt        # optional — Chatterbox TTS config
│   ├── openers.txt           # optional — opening lines, one per line
│   ├── gestures.txt          # optional — beat=gesture mappings
│   └── *.py                  # optional — custom tool implementations
└── persona_b/
    └── instructions.txt
```

`instructions.txt` is the only required file. All other files are optional; the app
falls back to built-in defaults (from the `default` profile or the tool registry)
when they are absent.

## Selecting a profile

At runtime, select a profile by name:

- **Admin UI**: Use the personality picker in the browser UI.
- **Environment variable**: `REACHY_MINI_CUSTOM_PROFILE=persona_a` in `.env`.
- **Startup settings**: Saved from the admin UI to `startup_settings.json`.
- **Locked profile**: `LOCKED_PROFILE` constant in `config.py` (overrides all others).

The profile name must match the subdirectory name exactly (case-sensitive on Linux/macOS).

## Persona template

A worked example of a well-structured persona is available at
`docs/persona_template/example/` in this repository. Use it as a starting point
when building new personas.
