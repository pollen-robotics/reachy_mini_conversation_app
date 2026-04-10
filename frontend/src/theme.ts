import { createTheme, type ThemeOptions } from "@mui/material/styles";

const shared: ThemeOptions = {
  typography: {
    fontFamily: '"Inter", "Helvetica Neue", Arial, sans-serif',
    h6: { fontWeight: 600 },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiCssBaseline: { styleOverrides: { body: { margin: 0, overflow: "hidden" } } },
    MuiButton: {
      defaultProps: { disableRipple: false },
      styleOverrides: {
        root: {
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          textTransform: "none" as const,
          fontWeight: 600,
          borderRadius: "10px",
          fontSize: "0.8125rem",
        },
        outlined: {
          borderColor: "#FF9500",
          color: "#FF9500",
          "&:hover": { borderColor: "#E08500", backgroundColor: "rgba(255, 149, 0, 0.08)" },
        },
      },
    },
    MuiPaper: { styleOverrides: { root: { backgroundImage: "none" } } },
    MuiCard: { styleOverrides: { root: { borderRadius: "10px" } } },
    MuiOutlinedInput: { styleOverrides: { root: { borderRadius: 10 } } },
    MuiCircularProgress: {
      styleOverrides: {
        root: { display: "block" },
        svg: { display: "block", overflow: "visible" },
        circle: { strokeLinecap: "round" },
      },
    },
    MuiToggleButton: { styleOverrides: { root: { textTransform: "none" as const, borderRadius: "8px" } } },
  },
};

export const lightTheme = createTheme({
  ...shared,
  palette: {
    mode: "light",
    primary: { main: "#FF9500", light: "#FFB340", dark: "#E08500", contrastText: "#fff" },
    secondary: { main: "#764ba2" },
    success: { main: "#22c55e" },
    error: { main: "#ef4444" },
    background: { default: "#fafafc", paper: "#fff" },
    divider: "rgba(0, 0, 0, 0.18)",
  },
  components: {
    ...shared.components,
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: "rgba(0, 0, 0, 0.9)",
          color: "#fff",
          fontSize: "11px",
          fontWeight: 500,
          padding: "10px 14px",
          borderRadius: "8px",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)",
          maxWidth: "300px",
          lineHeight: 1.6,
        },
        arrow: { color: "rgba(0, 0, 0, 0.9)" },
      },
    },
  },
});

export const darkTheme = createTheme({
  ...shared,
  palette: {
    mode: "dark",
    primary: { main: "#FF9500", light: "#FFB340", dark: "#E08500", contrastText: "#fff" },
    secondary: { main: "#764ba2" },
    success: { main: "#22c55e" },
    error: { main: "#ef4444" },
    background: { default: "#1a1a1a", paper: "#242424" },
    divider: "rgba(255, 255, 255, 0.12)",
  },
  components: {
    ...shared.components,
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          backgroundColor: "rgba(255, 255, 255, 0.95)",
          color: "#1d1d1f",
          fontSize: "11px",
          fontWeight: 500,
          padding: "10px 14px",
          borderRadius: "8px",
          boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
          maxWidth: "300px",
          lineHeight: 1.6,
        },
        arrow: { color: "rgba(255, 255, 255, 0.95)" },
      },
    },
  },
});
