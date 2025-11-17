"""Optimizer for ReachyMiniScript - further optimize IR for execution."""

from typing import List

from reachy_mini_conversation_app.rmscript.errors import Action, WaitAction, PictureAction, PlaySoundAction


class Optimizer:
    """Optimizes intermediate representation."""

    def optimize(self, ir: List[Action | WaitAction | PictureAction | PlaySoundAction]) -> List[Action | WaitAction | PictureAction | PlaySoundAction]:
        """Optimize IR.

        Current optimizations:
        - Merge consecutive wait actions
        - Remove no-op actions

        Future optimizations:
        - Combine compatible actions with same duration
        - Minimize movement time
        """
        optimized: List[Action | WaitAction | PictureAction | PlaySoundAction] = []

        i = 0
        while i < len(ir):
            action = ir[i]

            # Merge consecutive waits
            if isinstance(action, WaitAction):
                total_wait = action.duration
                j = i + 1
                while j < len(ir):
                    next_action = ir[j]
                    if isinstance(next_action, WaitAction):
                        total_wait += next_action.duration
                        j += 1
                    else:
                        break

                optimized.append(
                    WaitAction(
                        duration=total_wait,
                        source_line=action.source_line,
                        original_text=f"wait {total_wait}s",
                    )
                )
                i = j
                continue

            # Remove no-op actions (no actual movement)
            if isinstance(action, Action):
                if (
                    action.head_pose is None
                    and action.antennas is None
                    and action.body_yaw is None
                ):
                    # Skip this action - it does nothing
                    i += 1
                    continue

            optimized.append(action)
            i += 1

        return optimized
