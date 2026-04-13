import { useState, useRef, useCallback, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import ButtonBase from "@mui/material/ButtonBase";
import Fade from "@mui/material/Fade";
import { keyframes } from "@mui/material/styles";
import AddIcon from "@mui/icons-material/Add";
import { useTheme } from "@mui/material/styles";

const DOT_PATTERNS: Record<number, [number, number][]> = {
  1: [[1, 1]],
  2: [[0, 0], [2, 2]],
  3: [[0, 0], [1, 1], [2, 2]],
  4: [[0, 0], [2, 0], [0, 2], [2, 2]],
  5: [[0, 0], [2, 0], [1, 1], [0, 2], [2, 2]],
  6: [[0, 0], [2, 0], [0, 1], [2, 1], [0, 2], [2, 2]],
};

const GRID_POS = [0.25, 0.5, 0.75];

function DiceFace({ value, size, color }: { value: number; size: number; color: string }) {
  const dotSize = size * 0.16;
  const dots = DOT_PATTERNS[value] ?? DOT_PATTERNS[6];

  return (
    <Box
      sx={{
        width: size,
        height: size,
        borderRadius: `${size * 0.2}px`,
        border: `1.5px solid ${color}`,
        position: "relative",
        boxSizing: "border-box",
      }}
    >
      {dots.map(([col, row], i) => (
        <Box
          key={i}
          sx={{
            position: "absolute",
            width: dotSize,
            height: dotSize,
            borderRadius: "50%",
            bgcolor: color,
            left: `calc(${GRID_POS[col] * 100}% - ${dotSize / 2}px)`,
            top: `calc(${GRID_POS[row] * 100}% - ${dotSize / 2}px)`,
          }}
        />
      ))}
    </Box>
  );
}

import { BUILTIN_PROFILES } from "../config/builtinProfiles";
import type { BuiltinProfile } from "../config/builtinProfiles";
import ReachiesCarousel from "./ReachiesCarousel";

const pulseGlow = keyframes`
  0%, 100% { opacity: 0.3; transform: translate(-50%, -50%) scale(1); }
  50% { opacity: 0.45; transform: translate(-50%, -50%) scale(1.05); }
`;

const diceBounceIn = keyframes`
  0% { transform: scale(1) translateY(0); }
  20% { transform: scale(0.8) translateY(4px); }
  50% { transform: scale(1.2) translateY(-6px); }
  70% { transform: scale(0.95) translateY(1px); }
  100% { transform: scale(1) translateY(0); }
`;

const diceShake = keyframes`
  0%, 100% { transform: translateX(0) rotate(0deg); }
  20% { transform: translateX(-2px) rotate(-3deg); }
  40% { transform: translateX(2px) rotate(3deg); }
  60% { transform: translateX(-1.5px) rotate(-2deg); }
  80% { transform: translateX(1.5px) rotate(2deg); }
`;

interface Props {
  onSelect: (profile: BuiltinProfile) => void;
  onCreateCustom: () => void;
  onSkip: () => void;
}

type SpinPhase = "idle" | "entering" | "spinning";

const randDie = () => Math.floor(Math.random() * 6) + 1;

export default function WelcomeScreen({ onSelect, onCreateCustom, onSkip }: Props) {
  const [spinPhase, setSpinPhase] = useState<SpinPhase>("idle");
  const [highlightedIndex, setHighlightedIndex] = useState<number | null>(null);
  const [diceValues, setDiceValues] = useState({ d1: 4, d2: 2 });
  const spinTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isSpinning = spinPhase !== "idle";

  useEffect(() => {
    return () => {
      if (spinTimeoutRef.current) clearTimeout(spinTimeoutRef.current);
    };
  }, []);

  const handleRandom = useCallback(() => {
    if (spinPhase !== "idle") return;

    setSpinPhase("entering");
    setDiceValues({ d1: randDie(), d2: randDie() });

    const finalD1 = randDie();
    const finalD2 = randDie();
    const finalIndex = ((finalD1 + finalD2) - 1) % BUILTIN_PROFILES.length;
    const totalSteps = BUILTIN_PROFILES.length + finalIndex;

    let currentStep = 0;

    spinTimeoutRef.current = setTimeout(() => {
      setSpinPhase("spinning");

      const spin = () => {
        const currentIndex = currentStep % BUILTIN_PROFILES.length;
        setHighlightedIndex(currentIndex);

        const stepsRemaining = totalSteps - currentStep;
        if (stepsRemaining <= 0) {
          setDiceValues({ d1: finalD1, d2: finalD2 });
        } else {
          setDiceValues({ d1: randDie(), d2: randDie() });
        }

        currentStep++;

        if (currentStep <= totalSteps) {
          const progress = currentStep / totalSteps;
          const baseDelay = 40;
          const maxDelay = 200;
          const delay = baseDelay + (maxDelay - baseDelay) * Math.pow(progress, 4);
          spinTimeoutRef.current = setTimeout(spin, delay);
        } else {
          spinTimeoutRef.current = setTimeout(() => {
            setSpinPhase("idle");
            setHighlightedIndex(null);
            onSelect(BUILTIN_PROFILES[finalIndex]);
          }, 700);
        }
      };

      spin();
    }, 250);
  }, [spinPhase, onSelect]);

  return (
    <Box
      sx={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        bgcolor: "background.default",
        overflow: "auto",
      }}
    >
      {/* Hero with carousel */}
      <Fade in timeout={600}>
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            pt: { xs: 5, sm: 7 },
            pb: { xs: 4, sm: 5 },
            px: 3,
            flexShrink: 0,
          }}
        >
          <Box sx={{ position: "relative", mb: 1 }}>
            <Box
              sx={{
                position: "absolute",
                top: "50%",
                left: "50%",
                width: 100,
                height: 100,
                borderRadius: "50%",
                bgcolor: "primary.main",
                filter: "blur(50px)",
                animation: `${pulseGlow} 4s ease-in-out infinite`,
                pointerEvents: "none",
              }}
            />
            <ReachiesCarousel width={150} height={150} interval={1800} />
          </Box>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 800,
              mt: 1.5,
              textAlign: "center",
              fontSize: { xs: "1.6rem", sm: "1.85rem" },
            }}
          >
            Hey! I'm{" "}
            <Box component="span" sx={{ color: "primary.main" }}>
              Reachy Mini
            </Box>
            .
          </Typography>
          <Typography
            variant="body1"
            sx={{
              mt: 1,
              textAlign: "center",
              maxWidth: 400,
              color: "text.secondary",
              fontSize: "1rem",
            }}
          >
            Pick a <strong>personality</strong> and let's <strong>chat</strong>.
          </Typography>
        </Box>
      </Fade>

      {/* Personality grid */}
      <Fade in timeout={900}>
        <Box
          sx={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            px: { xs: 2, sm: 4 },
            pb: 4,
          }}
        >
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)" },
              gap: 1,
              maxWidth: 520,
              width: "100%",
            }}
          >
            <CreateCustomTile onSelect={onCreateCustom} disabled={isSpinning} />
            <RandomTile onSelect={handleRandom} spinPhase={spinPhase} diceValues={diceValues} />
            {BUILTIN_PROFILES.map((p, i) => (
              <PersonalityTile
                key={p.id}
                profile={p}
                onSelect={() => onSelect(p)}
                highlighted={highlightedIndex === i}
                disabled={isSpinning}
              />
            ))}
          </Box>

          <ButtonBase
            onClick={onSkip}
            disabled={isSpinning}
            sx={{
              mt: 4,
              mb: 2,
              px: 2,
              py: 1,
              borderRadius: 2,
              color: "text.secondary",
              fontSize: "0.85rem",
              "&:hover": { color: "text.primary", bgcolor: "action.hover" },
              transition: "all 0.15s ease",
              opacity: isSpinning ? 0.4 : 1,
            }}
          >
            Just start chatting &rarr;
          </ButtonBase>
        </Box>
      </Fade>
    </Box>
  );
}

function RandomTile({
  onSelect,
  spinPhase = "idle",
  diceValues,
}: {
  onSelect: () => void;
  spinPhase?: SpinPhase;
  diceValues: { d1: number; d2: number };
}) {
  const theme = useTheme();
  const isActive = spinPhase !== "idle";
  const isShaking = spinPhase === "spinning";
  const color = theme.palette.primary.main;

  const getDiceAnimation = (index: number) => {
    if (spinPhase === "entering")
      return `${diceBounceIn} 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) ${index * 0.06}s both`;
    if (spinPhase === "spinning")
      return `${diceShake} 0.1s ease-in-out infinite ${index * 0.03}s`;
    return "none";
  };

  return (
    <ButtonBase
      onClick={onSelect}
      disabled={isActive}
      sx={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        gap: 1.5,
        px: 1.5,
        py: 1.25,
        borderRadius: 1,
        border: 1,
        borderColor: isActive ? "primary.main" : "divider",
        bgcolor: "background.paper",
        transition: "all 0.2s ease",
        "&:hover": {
          borderColor: "primary.main",
          transform: isActive ? "none" : "translateY(-1px)",
        },
      }}
    >
      <Box
        sx={{
          width: 40,
          height: 40,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          gap: "4px",
        }}
      >
        <Box
          sx={{
            animation: getDiceAnimation(0),
            transform: "rotate(-8deg)",
            transition: "transform 0.2s ease",
          }}
        >
          <DiceFace value={diceValues.d1} size={18} color={isShaking ? color : theme.palette.text.secondary} />
        </Box>
        <Box
          sx={{
            animation: getDiceAnimation(1),
            transform: "rotate(8deg)",
            transition: "transform 0.2s ease",
          }}
        >
          <DiceFace value={diceValues.d2} size={18} color={isShaking ? color : theme.palette.text.secondary} />
        </Box>
      </Box>

      <Box sx={{ textAlign: "left", minWidth: 0, flex: 1 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.3, fontSize: "0.8rem" }}>
          {isActive ? "Rolling..." : "Surprise Me"}
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mt: 0.15, lineHeight: 1.3, fontSize: "0.7rem" }}
        >
          Random personality
        </Typography>
      </Box>
    </ButtonBase>
  );
}

function CreateCustomTile({ onSelect, disabled = false }: { onSelect: () => void; disabled?: boolean }) {
  return (
    <ButtonBase
      onClick={onSelect}
      disabled={disabled}
      sx={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        gap: 1.5,
        px: 1.5,
        py: 1.25,
        borderRadius: 1,
        border: 1,
        borderStyle: "dashed",
        borderColor: "divider",
        bgcolor: "background.paper",
        transition: "all 0.2s ease",
        opacity: disabled ? 0.4 : 1,
        "&:hover": {
          borderColor: disabled ? "divider" : "primary.main",
          transform: disabled ? "none" : "translateY(-1px)",
        },
      }}
    >
      <Box
        sx={{
          width: 36,
          height: 36,
          borderRadius: "50%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          border: "2px dashed",
          borderColor: "text.secondary",
          color: "text.secondary",
          flexShrink: 0,
        }}
      >
        <AddIcon sx={{ fontSize: 18 }} />
      </Box>
      <Box sx={{ textAlign: "left", minWidth: 0, flex: 1 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.3, fontSize: "0.8rem" }}>
          Create Your Own
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mt: 0.15, lineHeight: 1.3, fontSize: "0.7rem" }}
        >
          Custom personality
        </Typography>
      </Box>
    </ButtonBase>
  );
}

function PersonalityTile({
  profile,
  onSelect,
  highlighted = false,
  disabled = false,
}: {
  profile: BuiltinProfile;
  onSelect: () => void;
  highlighted?: boolean;
  disabled?: boolean;
}) {
  const { gradient } = profile.avatar;

  return (
    <ButtonBase
      onClick={onSelect}
      disabled={disabled}
      sx={{
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        gap: 1.5,
        px: 1.5,
        py: 1.25,
        borderRadius: 1,
        border: 2,
        borderColor: highlighted ? "primary.main" : "divider",
        bgcolor: "background.paper",
        transition: highlighted ? "all 0.06s ease-out" : "all 0.2s ease",
        transform: highlighted ? "scale(1.03)" : "scale(1)",
        boxShadow: highlighted ? (theme) => `0 0 20px ${theme.palette.primary.main}44, 0 4px 16px ${theme.palette.primary.main}22` : "none",
        zIndex: highlighted ? 2 : 1,
        "&:hover": {
          borderColor: disabled ? "divider" : "primary.main",
          transform: disabled ? "none" : highlighted ? "scale(1.03)" : "translateY(-1px)",
          boxShadow: disabled ? "none" : `0 4px 20px ${gradient[0]}22`,
        },
      }}
    >
      <Box sx={{ width: 40, height: 40, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
        <img
          src={profile.avatar.image}
          alt=""
          style={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            transition: highlighted ? "transform 0.06s ease-out" : "transform 0.2s ease",
            transform: highlighted ? "scale(1.1)" : "scale(1)",
          }}
        />
      </Box>
      <Box sx={{ textAlign: "left", minWidth: 0, flex: 1 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.3, fontSize: "0.8rem" }}>
          {profile.name}
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mt: 0.15, lineHeight: 1.3, fontSize: "0.7rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
        >
          {profile.description}
        </Typography>
      </Box>
    </ButtonBase>
  );
}

