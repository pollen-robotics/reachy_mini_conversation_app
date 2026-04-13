import { useState, useEffect } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";
import Drawer from "@mui/material/Drawer";
import IconButton from "@mui/material/IconButton";
import CloseIcon from "@mui/icons-material/Close";
import type { AppSettings, CustomProfile } from "../config/settings";
import { saveProfileToServer, applyProfileOnServer } from "../config/settings";
import { BUILTIN_PROFILES } from "../config/builtinProfiles";
import type { BuiltinProfile } from "../config/builtinProfiles";
import { ProfileCard, SavedCustomCard, NewCustomCard } from "./ProfileCard";
import { BuiltinProfileViewer, CustomProfileEditor } from "./ProfileModal";
import type { CustomModalSaveData } from "./ProfileModal";

interface Props {
  open: boolean;
  onClose: () => void;
  settings: AppSettings;
  onUpdate: (patch: Partial<AppSettings>) => void;
  initialCreate?: boolean;
}

type ModalTarget =
  | { kind: "builtin"; profile: BuiltinProfile }
  | { kind: "custom-new" }
  | { kind: "custom-edit"; profile: CustomProfile }
  | null;

function generateId(): string {
  return `custom_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export default function ProfileDrawer({ open, onClose, settings, onUpdate, initialCreate }: Props) {
  const [modal, setModal] = useState<ModalTarget>(() =>
    initialCreate ? { kind: "custom-new" } : null,
  );

  useEffect(() => {
    if (open && initialCreate) {
      setModal({ kind: "custom-new" });
    }
  }, [open, initialCreate]);

  const handleSelectBuiltin = (profile: BuiltinProfile) => {
    onUpdate({
      profileId: profile.id,
      voice: profile.voice,
      customInstructions: "",
    });
    applyProfileOnServer(profile.id, true);
    onClose();
  };

  const handleSelectCustom = (profile: CustomProfile) => {
    onUpdate({
      profileId: profile.id,
      voice: profile.voice,
      customInstructions: profile.instructions,
      customEnabledTools: profile.enabledTools,
    });
    const serverName = `user_personalities/${profile.name.replace(/\s+/g, "_")}`;
    applyProfileOnServer(serverName, true);
    onClose();
  };

  const handleDuplicateBuiltin = (profile: BuiltinProfile) => {
    const id = generateId();
    const newCustom: CustomProfile = {
      id,
      name: `${profile.name} (copy)`,
      instructions: profile.instructions,
      voice: profile.voice,
      enabledTools: [...profile.enabledTools],
    };
    const updatedProfiles = [...settings.customProfiles, newCustom];
    onUpdate({
      customProfiles: updatedProfiles,
      profileId: id,
      voice: newCustom.voice,
      customInstructions: newCustom.instructions,
      customEnabledTools: newCustom.enabledTools,
    });
    saveProfileToServer(newCustom);
    setModal({ kind: "custom-edit", profile: newCustom });
  };

  const handleSaveCustomModal = (data: CustomModalSaveData) => {
    if (data.id) {
      const updatedProfiles = settings.customProfiles.map((p) =>
        p.id === data.id
          ? { ...p, name: data.name, instructions: data.instructions, voice: data.voice, enabledTools: data.enabledTools }
          : p,
      );
      const isActive = settings.profileId === data.id;
      onUpdate({
        customProfiles: updatedProfiles,
        ...(isActive && {
          voice: data.voice,
          customInstructions: data.instructions,
          customEnabledTools: data.enabledTools,
        }),
      });
      saveProfileToServer({ id: data.id, name: data.name, instructions: data.instructions, voice: data.voice, enabledTools: data.enabledTools });
    } else {
      const id = generateId();
      const newProfile: CustomProfile = {
        id,
        name: data.name,
        instructions: data.instructions,
        voice: data.voice,
        enabledTools: data.enabledTools,
      };
      onUpdate({
        customProfiles: [...settings.customProfiles, newProfile],
        profileId: id,
        voice: newProfile.voice,
        customInstructions: newProfile.instructions,
        customEnabledTools: newProfile.enabledTools,
      });
      saveProfileToServer(newProfile);
    }
    setModal(null);
  };

  const handleDeleteCustom = (id: string) => {
    const updatedProfiles = settings.customProfiles.filter((p) => p.id !== id);
    const wasActive = settings.profileId === id;
    onUpdate({
      customProfiles: updatedProfiles,
      ...(wasActive && { profileId: "default", voice: "cedar", customInstructions: "", customEnabledTools: [] }),
    });
    setModal(null);
  };

  const isSubView = modal !== null;
  const drawerWidth = isSubView ? 520 : 420;

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      PaperProps={{
        sx: {
          width: { xs: "100%", sm: drawerWidth },
          maxWidth: "100vw",
          display: "flex",
          flexDirection: "column",
        },
      }}
    >
      {modal?.kind === "builtin" ? (
        <BuiltinProfileViewer
          profile={modal.profile}
          onBack={() => setModal(null)}
          onDuplicate={() => handleDuplicateBuiltin(modal.profile)}
        />
      ) : modal?.kind === "custom-new" || modal?.kind === "custom-edit" ? (
        <CustomProfileEditor
          editing={modal?.kind === "custom-edit" ? modal.profile : undefined}
          onBack={() => setModal(null)}
          onSave={handleSaveCustomModal}
          onDelete={
            modal?.kind === "custom-edit"
              ? () => handleDeleteCustom(modal.profile.id)
              : undefined
          }
        />
      ) : (
        <>
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
              Choose a Personality
            </Typography>
            <IconButton size="small" onClick={onClose}>
              <CloseIcon sx={{ fontSize: 20 }} />
            </IconButton>
          </Box>

          <Box sx={{ flex: 1, overflow: "auto", p: 2.5 }}>
            <Typography
              variant="overline"
              sx={{ display: "block", mb: 1.5, color: "text.secondary", fontWeight: 700, letterSpacing: 1.2, fontSize: "0.7rem" }}
            >
              Your Profiles
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                gap: 1,
                mb: 3,
              }}
            >
              <NewCustomCard onSelect={() => setModal({ kind: "custom-new" })} />
              {settings.customProfiles.map((cp) => (
                <SavedCustomCard
                  key={cp.id}
                  name={cp.name}
                  selected={settings.profileId === cp.id}
                  onSelect={() => handleSelectCustom(cp)}
                  onEdit={() => setModal({ kind: "custom-edit", profile: cp })}
                />
              ))}
            </Box>

            <Typography
              variant="overline"
              sx={{ display: "block", mb: 1.5, color: "text.secondary", fontWeight: 700, letterSpacing: 1.2, fontSize: "0.7rem" }}
            >
              Built-in Personalities
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                gap: 1,
              }}
            >
              {BUILTIN_PROFILES.map((p) => (
                <ProfileCard
                  key={p.id}
                  name={p.name}
                  description={p.description}
                  avatar={p.avatar}
                  selected={settings.profileId === p.id}
                  onSelect={() => handleSelectBuiltin(p)}
                  onDetails={() => setModal({ kind: "builtin", profile: p })}
                />
              ))}
            </Box>
          </Box>
        </>
      )}
    </Drawer>
  );
}
