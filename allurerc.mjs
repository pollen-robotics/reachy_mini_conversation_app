import { defineConfig } from "allure";

export default defineConfig({
  name: "Conversation App Tests",
  output: "./allure-report",
  historyPath: "./history/history.jsonl",
  plugins: {
    awesome: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "Conversation App Tests",
        singleFile: false,
      },
    },
  },
});
