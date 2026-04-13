import Box from "@mui/material/Box";
import ButtonBase from "@mui/material/ButtonBase";
import Typography from "@mui/material/Typography";
import IconButton from "@mui/material/IconButton";
import Tooltip from "@mui/material/Tooltip";
import SettingsOutlinedIcon from "@mui/icons-material/SettingsOutlined";
import RefreshIcon from "@mui/icons-material/Refresh";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import type { BuiltinProfile } from "../config/builtinProfiles";
import type { CustomProfile } from "../config/settings";

interface HeaderProps {
  builtinProfile: BuiltinProfile | null;
  customProfile: CustomProfile | null;
  onOpenProfiles: () => void;
  onOpenSettings: () => void;
  onReload?: () => void;
  isConnected: boolean;
}

export default function Header({
  builtinProfile,
  customProfile,
  onOpenProfiles,
  onOpenSettings,
  onReload,
  isConnected,
}: HeaderProps) {
  const name = builtinProfile?.name ?? customProfile?.name ?? "Default";
  const image = builtinProfile?.avatar.image;
  const initial = customProfile?.name?.trim()?.[0]?.toUpperCase() ?? "R";

  return (
    <Box
      sx={{
        px: 1.5,
        py: 0.75,
        display: "flex",
        alignItems: "center",
        gap: 1,
        borderBottom: "1px solid",
        borderColor: (t) => t.palette.mode === "dark" ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.06)",
        minHeight: 52,
        flexShrink: 0,
      }}
    >
      {/* Profile identity — clickable to open drawer */}
      <ButtonBase
        onClick={onOpenProfiles}
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1.25,
          borderRadius: 2,
          px: 1,
          py: 0.5,
          mr: "auto",
          minWidth: 0,
          "&:hover": { bgcolor: "action.hover" },
          transition: "background-color 0.15s ease",
        }}
      >
        {image ? (
          <img src={image} alt="" style={{ width: 38, height: 38, objectFit: "contain", flexShrink: 0 }} />
        ) : (
          <Box
            sx={{
              width: 38,
              height: 38,
              borderRadius: "50%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              border: "2px solid",
              borderColor: "primary.main",
              color: "primary.main",
              fontWeight: 700,
              fontSize: "0.9rem",
              flexShrink: 0,
            }}
          >
            {initial}
          </Box>
        )}

        <Typography
          variant="subtitle2"
          sx={{
            fontWeight: 700,
            fontSize: "0.88rem",
            lineHeight: 1.2,
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            maxWidth: 140,
          }}
        >
          {name}
        </Typography>
        <KeyboardArrowDownIcon sx={{ fontSize: 18, color: "text.secondary", ml: -0.5 }} />
      </ButtonBase>

      {isConnected && onReload && (
        <Tooltip title="New session" arrow>
          <IconButton size="small" onClick={onReload} aria-label="Start new session">
            <RefreshIcon sx={{ fontSize: 20 }} />
          </IconButton>
        </Tooltip>
      )}

      <IconButton size="small" onClick={onOpenSettings} sx={{ ml: 0.5 }}>
        <SettingsOutlinedIcon sx={{ fontSize: 20 }} />
      </IconButton>
    </Box>
  );
}
