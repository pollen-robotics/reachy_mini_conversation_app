# Reachy Mini Astronomer Buddy

This demo shows how to use Reachy Mini as an astronomer buddy that can help you find stars and constellations in the night sky.

## Configuration

- For now you have to initially orient Reachy Mini in the direction of the North.
- The local latitude and longitude are hardcoded in astronomer.py, you can change them to your own location.

## Requirements

In `pyproject.toml` add the following to the `extra` section:
```
astronomer = ["rapidfuzz", "astropy"]
```

Then run
```
uv sync --extra astronomer   
```

## Tools

Reachy Mini has access to the following tools:
- `look_at_astronomical_object` : Point Reachy Mini's head towards a specified astronomical object (star, planet, constellation, etc.) given its name.
- `what_is_visible_now` : List the astronomical objects that are currently visible from Reachy Mini's location.