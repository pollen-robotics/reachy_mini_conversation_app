---
config:
  layout: dagre
  flowchart:
    htmlLabels: true
---
flowchart TB
    User(["<span style='font-size:16px;font-weight:bold;'>User</span><br><span style='font-size:13px;color:#01579b;'>Person interacting with system</span>"]) 
      -- audio stream --> 
    UI@{ label: "<span style='font-size:16px;font-weight:bold;'>UI Layer</span><br><span style='font-size:13px;color:#0277bd;'>Gradio/Console</span>" }

    UI -- audio stream --> 
    OpenAI@{ label: "<span style='font-size:17px;font-weight:bold;'>gpt-realtime API</span><br><span style='font-size:13px; color:#7b1fa2;'>Audio+Tool Calls+Vision</span>" }

    OpenAI -- audio stream --> 
    Motion@{ label: "<span style='font-size:16px;font-weight:bold;'>Motion Control</span><br><span style='font-size:13px;color:#f57f17;'>Audio Sync + Tracking</span>" }
    
    OpenAI -- tool calls --> 
    Handlers@{ label: "<span style='font-size:16px;font-weight:bold;'>Tool Handlers</span><br><span style='font-size:12px;color:#f9a825;'>move_head, camera, head_tracking,<br/>dance, play_emotion, do_nothing</span>" }

    Handlers -- movement 
    requests --> Motion

    Handlers -- camera frames, face tracking --> 
    Camera@{ label: "<span style='font-size:16px;font-weight:bold;'>Camera Worker</span><br><span style='font-size:13px;color:#f57f17;'>Frame Buffer + Face Tracking</span>" }
    
    Handlers -. image for 
    analysis .-> OpenAI

    Camera -- face tracking --> Motion

    Camera -. frames .-> 
    Vision@{ label: "<span style='font-size:16px;font-weight:bold;'>Vision Processor</span><br><span style='font-size:13px;color:#7b1fa2;'>Local VLM (optional)</span>" }

    Vision -. description .-> Handlers

    Robot@{ label: "<span style='font-size:16px;font-weight:bold;'>reachy_mini</span><br><span style='font-size:13px;color:#c62828;'>Robot Control Library</span>" } 
    -- camera 
    frames --> Camera

    Motion -- commands --> Robot

    Handlers -- results --> OpenAI

     User:::userStyle
     UI:::uiStyle
     OpenAI:::aiStyle
     Motion:::coreStyle
     Handlers:::toolStyle
     Camera:::coreStyle
     Vision:::aiStyle
     Robot:::hardwareStyle
    classDef userStyle fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef uiStyle fill:#b3e5fc,stroke:#0277bd,stroke-width:2px
    classDef aiStyle fill:#e1bee7,stroke:#7b1fa2,stroke-width:3px
    classDef coreStyle fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    classDef hardwareStyle fill:#ef9a9a,stroke:#c62828,stroke-width:3px
    classDef toolStyle fill:#fffde7,stroke:#f9a825,stroke-width:1px
