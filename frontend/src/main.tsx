import React, { useMemo } from "react";
import ReactDOM from "react-dom/client";
import CssBaseline from "@mui/material/CssBaseline";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import useMediaQuery from "@mui/material/useMediaQuery";
import { lightTheme, darkTheme } from "./theme";
import App from "./App";

function useThemeFromContext() {
  const params = useMemo(() => new URLSearchParams(window.location.search), []);
  const explicitTheme = params.get("theme");
  const bgParam = params.get("bg");
  const systemDark = useMediaQuery("(prefers-color-scheme: dark)");

  const isDark =
    explicitTheme === "dark" ? true
    : explicitTheme === "light" ? false
    : systemDark;

  const base = isDark ? darkTheme : lightTheme;

  if (!bgParam) return base;

  return createTheme(base, {
    palette: { background: { default: `#${bgParam}` } },
  });
}

function Root() {
  const theme = useThemeFromContext();

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
