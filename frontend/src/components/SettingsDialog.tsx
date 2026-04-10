import { useState, useEffect, useRef } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import Button from "@mui/material/Button";
import Drawer from "@mui/material/Drawer";
import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import Divider from "@mui/material/Divider";
import Collapse from "@mui/material/Collapse";
import Fade from "@mui/material/Fade";
import Switch from "@mui/material/Switch";
import CloseIcon from "@mui/icons-material/Close";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import VisibilityIcon from "@mui/icons-material/Visibility";
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import type { AppSettings } from "../config/settings";

interface Props {
  open: boolean;
  onClose: () => void;
  settings: AppSettings;
  onUpdate: (patch: Partial<AppSettings>) => void;
  hasKey: boolean;
}

function StatusChip({ ok, label }: { ok: boolean; label: string }) {
  return (
    <Box
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: 0.75,
        px: 1.25,
        py: 0.375,
        borderRadius: 10,
        bgcolor: ok ? "success.main" : "text.disabled",
        color: "#fff",
        fontSize: "0.65rem",
        fontWeight: 700,
        letterSpacing: "0.03em",
        textTransform: "uppercase",
      }}
    >
      {label}
    </Box>
  );
}

export default function SettingsDialog({ open, onClose, settings, onUpdate, hasKey }: Props) {
  const [keyInput, setKeyInput] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [saved, setSaved] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const savedTimerRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (!open) {
      setKeyInput("");
      setShowKey(false);
      setSaved(false);
      setConfirmRemove(false);
    }
  }, [open]);

  useEffect(() => () => clearTimeout(savedTimerRef.current), []);

  const handleSaveKey = () => {
    const key = keyInput.trim();
    if (!key) return;
    onUpdate({ openaiApiKey: key });
    setKeyInput("");
    setShowKey(false);
    setSaved(true);
    clearTimeout(savedTimerRef.current);
    savedTimerRef.current = setTimeout(() => setSaved(false), 2500);
  };

  const handleClearKey = () => {
    if (!confirmRemove) {
      setConfirmRemove(true);
      return;
    }
    onUpdate({ openaiApiKey: "" });
    setKeyInput("");
    setConfirmRemove(false);
  };

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      PaperProps={{ sx: { width: { xs: "100%", sm: 420 }, maxWidth: "100vw" } }}
    >
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          px: 2.5,
          py: 2,
          borderBottom: 1,
          borderColor: "divider",
          flexShrink: 0,
        }}
      >
        <Typography variant="h6" sx={{ fontWeight: 700, flex: 1, fontSize: "1rem" }}>
          Settings
        </Typography>
        <IconButton size="small" onClick={onClose}>
          <CloseIcon sx={{ fontSize: 20 }} />
        </IconButton>
      </Box>

      {/* Body */}
      <Box sx={{ flex: 1, overflow: "auto", p: 2.5, display: "flex", flexDirection: "column", gap: 3 }}>
        {/* API Key */}
        <Box>
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                OpenAI API Key
              </Typography>
              <StatusChip ok={hasKey} label={hasKey ? "Configured" : "Required"} />
            </Box>

            <TextField
              fullWidth
              size="small"
              type={showKey ? "text" : "password"}
              placeholder={hasKey ? "Enter new key to update" : "sk-proj-..."}
              value={keyInput}
              onChange={(e) => setKeyInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSaveKey()}
              sx={{ mb: 1.5 }}
              slotProps={{
                input: {
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        size="small"
                        onClick={() => setShowKey((v) => !v)}
                        edge="end"
                        tabIndex={-1}
                        sx={{ color: "text.disabled" }}
                      >
                        {showKey ? (
                          <VisibilityOffIcon sx={{ fontSize: 16 }} />
                        ) : (
                          <VisibilityIcon sx={{ fontSize: 16 }} />
                        )}
                      </IconButton>
                    </InputAdornment>
                  ),
                },
              }}
            />

            <Button
              fullWidth
              variant="contained"
              onClick={handleSaveKey}
              disabled={!keyInput.trim()}
              disableElevation
              sx={{ fontWeight: 600, fontSize: "0.8rem", mb: 1.5 }}
            >
              {hasKey ? "Update Key" : "Save Key"}
            </Button>

            <Collapse in={saved}>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 0.75,
                  mb: 1.5,
                }}
              >
                <CheckCircleOutlineIcon sx={{ fontSize: 16, color: "success.main" }} />
                <Typography variant="caption" sx={{ color: "success.main", fontWeight: 600, fontSize: "0.75rem" }}>
                  API key saved successfully
                </Typography>
              </Box>
            </Collapse>

            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <Box
                component="a"
                href="https://platform.openai.com/api-keys"
                target="_blank"
                rel="noopener noreferrer"
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 0.5,
                  color: "text.secondary",
                  textDecoration: "none",
                  fontSize: "0.75rem",
                  "&:hover": { color: "primary.main" },
                  transition: "color 0.15s ease",
                }}
              >
                <OpenInNewIcon sx={{ fontSize: 14 }} />
                {hasKey ? "Manage keys on OpenAI" : "Get a key from OpenAI"}
              </Box>

              {hasKey && !saved && (
                <>
                  {!confirmRemove ? (
                    <Button
                      size="small"
                      color="error"
                      variant="text"
                      onClick={handleClearKey}
                      sx={{ fontSize: "0.75rem", minHeight: 0, py: 0.25, px: 1, textTransform: "none" }}
                    >
                      Remove
                    </Button>
                  ) : (
                    <Fade in>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                        <Typography variant="caption" sx={{ color: "error.main", fontSize: "0.72rem", fontWeight: 500 }}>
                          Sure?
                        </Typography>
                        <Button
                          size="small"
                          color="error"
                          variant="contained"
                          disableElevation
                          onClick={handleClearKey}
                          sx={{ fontSize: "0.7rem", minHeight: 0, py: 0.25, px: 1, minWidth: 0 }}
                        >
                          Yes
                        </Button>
                        <Button
                          size="small"
                          variant="text"
                          color="inherit"
                          onClick={() => setConfirmRemove(false)}
                          sx={{ fontSize: "0.7rem", minHeight: 0, py: 0.25, px: 1, minWidth: 0, color: "text.secondary" }}
                        >
                          No
                        </Button>
                      </Box>
                    </Fade>
                  )}
                </>
              )}
            </Box>
          </Box>

        <Divider />

        {/* Robot */}
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              Robot
            </Typography>

            {/* Connection URL */}
            <Box>
              <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 1.5 }}>
                <Typography variant="body2" sx={{ fontWeight: 500, fontSize: "0.82rem" }}>
                  Connection
                </Typography>
                <StatusChip ok label={settings.daemonUrl || "Auto"} />
              </Box>
              <TextField
                fullWidth
                size="small"
                placeholder="http://reachy-mini.local:8000"
                value={settings.daemonUrl}
                onChange={(e) => onUpdate({ daemonUrl: e.target.value })}
              />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 1, display: "block", lineHeight: 1.5, fontSize: "0.72rem" }}
              >
                Leave empty to auto-detect. Set to the robot IP when running locally.
              </Typography>
            </Box>

            {/* Camera toggle */}
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <Box>
                <Typography variant="body2" sx={{ fontWeight: 500, fontSize: "0.82rem" }}>
                  Camera
                </Typography>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: "block", lineHeight: 1.5, fontSize: "0.72rem" }}
                >
                  Enable the video feed for visual interactions
                </Typography>
              </Box>
              <Switch
                checked={settings.cameraEnabled}
                onChange={(e) => onUpdate({ cameraEnabled: e.target.checked })}
                size="small"
              />
            </Box>
          </Box>
      </Box>

      {/* Footer */}
    </Drawer>
  );
}
