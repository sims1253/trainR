"use client";

import { useSyncExternalStore } from "react";

interface ClientOnlyProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

/**
 * Empty subscribe function for useSyncExternalStore.
 * Since we're only detecting client-side rendering, there's nothing to subscribe to.
 */
const emptySubscribe = () => () => {};

/**
 * ClientOnly component for handling SSR-safe rendering of client-only content.
 *
 * Uses useSyncExternalStore to detect client-side rendering without triggering
 * the react-hooks/set-state-in-effect ESLint rule. This is the recommended
 * React pattern for conditional client-side rendering.
 *
 * Use this wrapper for components that should only render on the client side,
 * such as Recharts with ResponsiveContainer which causes hydration issues.
 */
export function ClientOnly({ children, fallback = null }: ClientOnlyProps) {
  const hasMounted = useSyncExternalStore(
    emptySubscribe,
    () => true, // Client snapshot: always true
    () => false, // Server snapshot: always false
  );

  if (!hasMounted) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
}
