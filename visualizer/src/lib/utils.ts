import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format a decimal as a percentage string
 * @param value - Decimal value (0-1)
 * @returns Formatted percentage string (e.g., "33.3%")
 */
export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
