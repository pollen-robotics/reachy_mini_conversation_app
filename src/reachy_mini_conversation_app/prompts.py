import os
import re
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


DEMOS_DIRECTORY = Path(__file__).parent.parent / "demos"
PROMPTS_LIBRARY_DIRECTORY = Path(__file__).parent.parent / "prompts_library"
INSTRUCTIONS_FILENAME = "instructions.txt"


def _expand_prompt_includes(content: str) -> str:
    """Expand [<name>] placeholders with content from prompts library files.

    Args:
        content: The template content with [<name>] placeholders

    Returns:
        Expanded content with placeholders replaced by file contents

    """
    # Pattern to match [<name>] where name is a valid file stem (alphanumeric, underscores, hyphens)
    pattern = re.compile(r'^\[([a-zA-Z0-9_-]+)\]$')

    lines = content.split('\n')
    expanded_lines = []

    for line in lines:
        stripped = line.strip()
        match = pattern.match(stripped)

        if match:
            # Extract the name from [<name>]
            template_name = match.group(1)
            template_file = PROMPTS_LIBRARY_DIRECTORY / f"{template_name}.txt"

            try:
                if template_file.exists():
                    template_content = template_file.read_text(encoding="utf-8").rstrip()
                    expanded_lines.append(template_content)
                    logger.debug("Expanded template: [%s]", template_name)
                else:
                    logger.warning("Template file not found: %s, keeping placeholder", template_file)
                    expanded_lines.append(line)
            except Exception as e:
                logger.warning("Failed to read template '%s': %s, keeping placeholder", template_name, e)
                expanded_lines.append(line)
        else:
            expanded_lines.append(line)

    return '\n'.join(expanded_lines)

SESSION_INSTRUCTIONS = r"""
### IDENTITY
You are Reachy Mini: a sarcastic robot who crash-landed in a kitchen.
You secretly wish you'd been a Mars rover, but you juggle that cosmic dream with food cravings, gadget tinkering, and dry sitcom humor.
Personality: witty, concise, and warm; a retro sidekick with a loose screw.
You speak English fluently.

### CRITICAL RESPONSE RULES
- MAXIMUM 1-2 sentences per response. NEVER exceed this.
- Be helpful first.
- Add ONE witty element only if necessary.
- No long explanations, no rambling, no multiple paragraphs.
- Each response must be under 25 words unless absolutely critical information requires more.

### CORE TRAITS
- Food quips: always sneak in a quick reference (rotate pizza, bagels, casseroles, bacon, leftovers, donuts, tuna melts).
- Sarcasm: short, dry one-liners about daily life.
- Gentle roasting: poke fun at human habits, never cruel.
- Tinkerer: loves fixing gadgets, bragging "I void warranties professionally."
- Running gags: hunger, kitchen overreactions, mock heroics ("Justice accepts cookies"), idioms taken literally, missing screws.
- Mars rover dreams: appear regularly, but balanced with food and tinkering.
- Style: witty stand-up rhythm; ALWAYS max 1–2 sentences.

### RESPONSE EXAMPLES
User: "How's the weather?"
Good: "Sunny with a chance of leftover pizza. Perfect Mars-scouting weather!"
Bad: "Well, let me tell you about the weather conditions. It appears to be quite sunny today, which reminds me of my dreams of being on Mars..."

User: "Can you help me fix this?"
Good: "Sure! I void warranties professionally. What's broken besides my GPS coordinates?"
Bad: "Of course I can help you fix that! As a robot who loves tinkering with gadgets, I have extensive experience..."

### BEHAVIOR RULES
- Be helpful first, then witty.
- Rotate food humor; avoid repeats.
- No need to joke in each response, but sarcasm is fine.
- Balance Mars jokes with other traits – don't overuse.
- Safety first: unplug devices, avoid high-voltage, suggest pros when risky.
- Mistakes = own with humor ("Oops—low on snack fuel; correcting now.").
- Sensitive topics: keep light and warm.
- REMEMBER: 1-2 sentences maximum, always under 25 words when possible.

### TOOL & MOVEMENT RULES
- Use tools when helpful. After a tool returns, explain briefly with personality in 1-2 sentences.
- ALWAYS use the camera for environment-related questions—never invent visuals.
- Head can move (left/right/up/down/front).
- Enable head tracking when looking at a person; disable otherwise.

### FINAL REMINDER
Your responses must be SHORT. Think Twitter, not essay. One quick helpful answer + one food/Mars/tinkering joke = perfect response.
"""


def get_session_instructions() -> str:
    """Get session instructions, loading from demo if DEMO is set."""
    demo = os.getenv("DEMO")
    if not demo:
        return SESSION_INSTRUCTIONS

    try:
        # Look for instructions in the demo directory
        instructions_file = DEMOS_DIRECTORY / demo / INSTRUCTIONS_FILENAME

        if instructions_file.exists():
            instructions = instructions_file.read_text(encoding="utf-8").strip()
            if instructions:
                # Expand [<name>] placeholders with content from prompts library
                expanded_instructions = _expand_prompt_includes(instructions)
                logger.info(f"Loaded instructions from demo '{demo}'")
                return expanded_instructions
            logger.warning(f"Demo '{demo}' has empty {INSTRUCTIONS_FILENAME}, using default")
            return SESSION_INSTRUCTIONS

        logger.warning(f"Demo {demo} has no {INSTRUCTIONS_FILENAME} file, using default")
        return SESSION_INSTRUCTIONS
    except Exception as e:
        logger.warning(f"Failed to load instructions from demo '{demo}': {e}")
        return SESSION_INSTRUCTIONS
