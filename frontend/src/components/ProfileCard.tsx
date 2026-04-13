import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Card from "@mui/material/Card";
import CardActionArea from "@mui/material/CardActionArea";
import IconButton from "@mui/material/IconButton";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import EditOutlinedIcon from "@mui/icons-material/EditOutlined";
import AddIcon from "@mui/icons-material/Add";
import type { ProfileAvatar } from "../config/builtinProfiles";

interface Props {
  name: string;
  description: string;
  avatar: ProfileAvatar;
  selected: boolean;
  onSelect: () => void;
  onDetails: () => void;
}

const cardSx = (selected: boolean) => ({
  position: "relative" as const,
  display: "flex",
  flexDirection: "row" as const,
  borderColor: selected ? "primary.main" : "divider",
  borderWidth: selected ? 2 : 1,
  transition: "all 0.2s ease",
  "&:hover": {
    borderColor: selected ? "primary.main" : "text.secondary",
    "& .details-btn": { opacity: 1 },
  },
});

export function ProfileCard({ name, description, avatar, selected, onSelect, onDetails }: Props) {
  return (
    <Card variant="outlined" sx={cardSx(selected)}>
      <CardActionArea
        onClick={onSelect}
        sx={{ px: 1.5, py: 1.25, display: "flex", flexDirection: "row", alignItems: "center", gap: 1.5, flex: 1 }}
      >
        <Box
          sx={{
            width: 40,
            height: 40,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
          }}
        >
          <img
            src={avatar.image}
            alt=""
            style={{ width: "100%", height: "100%", objectFit: "contain" }}
          />
        </Box>
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.3, fontSize: "0.8rem" }}>
            {name}
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: "block", mt: 0.15, lineHeight: 1.3, fontSize: "0.7rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}
          >
            {description}
          </Typography>
        </Box>
      </CardActionArea>

      {selected && (
        <CheckCircleIcon
          color="primary"
          sx={{ position: "absolute", top: 6, left: 6, fontSize: 14 }}
        />
      )}

      <IconButton
        className="details-btn"
        size="small"
        onClick={(e) => { e.stopPropagation(); onDetails(); }}
        sx={{
          position: "absolute",
          top: 4,
          right: 4,
          opacity: selected ? 0.7 : 0,
          transition: "opacity 0.15s ease",
          bgcolor: "background.paper",
          "&:hover": { bgcolor: "action.hover", opacity: 1 },
        }}
      >
        <InfoOutlinedIcon sx={{ fontSize: 14 }} />
      </IconButton>
    </Card>
  );
}

interface SavedCustomProps {
  name: string;
  selected: boolean;
  onSelect: () => void;
  onEdit: () => void;
}

export function SavedCustomCard({ name, selected, onSelect, onEdit }: SavedCustomProps) {
  const initial = name.trim() ? name.trim()[0].toUpperCase() : "?";
  return (
    <Card variant="outlined" sx={cardSx(selected)}>
      <CardActionArea
        onClick={onSelect}
        sx={{ px: 1.5, py: 1.25, display: "flex", flexDirection: "row", alignItems: "center", gap: 1.5, flex: 1 }}
      >
        <Box
          sx={{
            width: 36,
            height: 36,
            borderRadius: "50%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            border: "2px solid",
            borderColor: "primary.main",
            color: "primary.main",
            fontSize: "0.85rem",
            fontWeight: 700,
            flexShrink: 0,
          }}
        >
          {initial}
        </Box>
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.3, fontSize: "0.8rem" }}>
            {name}
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: "block", mt: 0.15, lineHeight: 1.3, fontSize: "0.7rem" }}
          >
            Custom profile
          </Typography>
        </Box>
      </CardActionArea>

      {selected && (
        <CheckCircleIcon
          color="primary"
          sx={{ position: "absolute", top: 6, left: 6, fontSize: 14 }}
        />
      )}

      <IconButton
        className="details-btn"
        size="small"
        onClick={(e) => { e.stopPropagation(); onEdit(); }}
        sx={{
          position: "absolute",
          top: 4,
          right: 4,
          opacity: selected ? 0.7 : 0,
          transition: "opacity 0.15s ease",
          bgcolor: "background.paper",
          "&:hover": { bgcolor: "action.hover", opacity: 1 },
        }}
      >
        <EditOutlinedIcon sx={{ fontSize: 14 }} />
      </IconButton>
    </Card>
  );
}

interface NewCustomProps {
  onSelect: () => void;
}

export function NewCustomCard({ onSelect }: NewCustomProps) {
  return (
    <Card
      variant="outlined"
      sx={{
        position: "relative",
        borderColor: "divider",
        borderWidth: 1,
        borderStyle: "dashed",
        transition: "all 0.2s ease",
        display: "flex",
        flexDirection: "row",
        "&:hover": { borderColor: "text.secondary" },
      }}
    >
      <CardActionArea
        onClick={onSelect}
        sx={{ px: 1.5, py: 1.25, display: "flex", flexDirection: "row", alignItems: "center", gap: 1.5, flex: 1 }}
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
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.3, fontSize: "0.8rem" }}>
            New Profile
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: "block", mt: 0.15, lineHeight: 1.3, fontSize: "0.7rem" }}
          >
            Create a personality
          </Typography>
        </Box>
      </CardActionArea>
    </Card>
  );
}
