"use client";

import Image from "next/image";
import { getTeamColor } from "@/lib/team-colors";

interface TeamLogoProps {
  team: string;
  size?: number;
  className?: string;
}

export function TeamLogo({ team, size = 32, className = "" }: TeamLogoProps) {
  const { primary, abbr, logo } = getTeamColor(team);

  if (!logo) {
    return (
      <div
        className={`rounded-full flex items-center justify-center text-[10px] font-bold text-white ${className}`}
        style={{ backgroundColor: primary, width: size, height: size }}
      >
        {abbr}
      </div>
    );
  }

  return (
    <div
      className={`relative flex items-center justify-center ${className}`}
      style={{ width: size, height: size }}
    >
      <Image
        src={logo}
        alt={`${team} logo`}
        width={size}
        height={size}
        className="object-contain"
      />
    </div>
  );
}
