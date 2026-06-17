export const TEAM_COLORS: Record<string, { primary: string; secondary: string; abbr: string; logo: string }> = {
  "Adelaide": { primary: "#002B5C", secondary: "#FFD700", abbr: "ADE", logo: "/teams/Adelaide.png" },
  "Brisbane Lions": { primary: "#A30046", secondary: "#0055A3", abbr: "BRL", logo: "/teams/Brisbane.png" },
  "Carlton": { primary: "#0E1E2D", secondary: "#FFFFFF", abbr: "CAR", logo: "/teams/Carlton.png" },
  "Collingwood": { primary: "#000000", secondary: "#FFFFFF", abbr: "COL", logo: "/teams/Collingwood.png" },
  "Essendon": { primary: "#CC2031", secondary: "#000000", abbr: "ESS", logo: "/teams/Essendon.png" },
  "Fremantle": { primary: "#2A0D45", secondary: "#FFFFFF", abbr: "FRE", logo: "/teams/Fremantle.png" },
  "Geelong": { primary: "#001F3D", secondary: "#FFFFFF", abbr: "GEE", logo: "/teams/Geelong.png" },
  "Gold Coast": { primary: "#D4001E", secondary: "#FFD700", abbr: "GCS", logo: "/teams/GoldCoast.png" },
  "Greater Western Sydney": { primary: "#F47920", secondary: "#4A4A4A", abbr: "GWS", logo: "/teams/Giants.png" },
  "Hawthorn": { primary: "#4D2004", secondary: "#FBBF24", abbr: "HAW", logo: "/teams/Hawthorn.png" },
  "Melbourne": { primary: "#0F1131", secondary: "#CC2031", abbr: "MEL", logo: "/teams/Melbourne.png" },
  "North Melbourne": { primary: "#003C71", secondary: "#FFFFFF", abbr: "NME", logo: "/teams/NorthMelbourne.png" },
  "Port Adelaide": { primary: "#008AAB", secondary: "#000000", abbr: "PTA", logo: "/teams/PortAdelaide.png" },
  "Richmond": { primary: "#000000", secondary: "#FED102", abbr: "RIC", logo: "/teams/Richmond.png" },
  "St Kilda": { primary: "#ED1C24", secondary: "#000000", abbr: "STK", logo: "/teams/StKilda.png" },
  "Sydney": { primary: "#ED171F", secondary: "#FFFFFF", abbr: "SYD", logo: "/teams/Sydney.png" },
  "West Coast": { primary: "#002B5C", secondary: "#F2A900", abbr: "WCE", logo: "/teams/WestCoast.png" },
  "Western Bulldogs": { primary: "#014896", secondary: "#CC2031", abbr: "WBD", logo: "/teams/Bulldogs.png" },
};

export function getTeamColor(team: string) {
  return TEAM_COLORS[team] ?? { primary: "#6b7280", secondary: "#9ca3af", abbr: team.slice(0, 3).toUpperCase(), logo: "" };
}
