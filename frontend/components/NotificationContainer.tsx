"use client";

import { useStore } from "@/store";
import { NotificationToast } from "@/components/ui";

export function NotificationContainer() {
  const { notifications, dismissNotification } = useStore((s) => ({
    notifications: s.notifications,
    dismissNotification: s.dismissNotification,
  }));

  if (notifications.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2">
      {notifications.map((n) => (
        <NotificationToast
          key={n.id}
          notification={n}
          onDismiss={() => dismissNotification(n.id)}
        />
      ))}
    </div>
  );
}
