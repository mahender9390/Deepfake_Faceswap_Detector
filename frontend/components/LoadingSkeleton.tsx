"use client";

// LoadingSkeleton — shimmer placeholder shown while the API is processing.

export default function LoadingSkeleton() {
  return (
    <div className="w-full rounded-2xl p-6 space-y-5 glass">
      {/* Badge placeholder */}
      <div className="flex flex-col items-center gap-5">
        <div className="skeleton h-12 w-32 rounded-full" />
        <div className="skeleton rounded-full" style={{ width: 160, height: 160 }} />
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3">
        {[...Array(2)].map((_, i) => (
          <div key={i} className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.06] space-y-2">
            <div className="skeleton h-3 w-16 rounded" />
            <div className="skeleton h-5 w-24 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}
