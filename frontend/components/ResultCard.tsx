"use client";

import { motion } from "framer-motion";
import { VideoResult } from "@/lib/api";
import { ShieldCheck, ShieldAlert, Clock, Crosshair, Film } from "lucide-react";

interface Props { result: VideoResult; }

function ConfidenceRing({ value, isFake }: { value: number; isFake: boolean }) {
  const SIZE = 160, STROKE = 10, RADIUS = (SIZE - STROKE) / 2;
  const CIRC = 2 * Math.PI * RADIUS;
  const color = isFake ? "#EF4444" : "#22C55E";
  return (
    <div className="relative" style={{ width: SIZE, height: SIZE }}>
      <svg width={SIZE} height={SIZE} className="-rotate-90">
        <circle cx={SIZE/2} cy={SIZE/2} r={RADIUS} fill="none"
          stroke="rgba(255,255,255,0.06)" strokeWidth={STROKE} />
        <motion.circle cx={SIZE/2} cy={SIZE/2} r={RADIUS} fill="none"
          stroke={color} strokeWidth={STROKE} strokeLinecap="round"
          strokeDasharray={CIRC}
          initial={{ strokeDashoffset: CIRC }}
          animate={{ strokeDashoffset: CIRC - (value / 100) * CIRC }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          style={{ filter: `drop-shadow(0 0 6px ${color}88)` }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.p initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, type: "spring" }}
          className="text-3xl font-bold" style={{ color }}>
          {value.toFixed(1)}%
        </motion.p>
        <p className="text-[10px] text-zinc-500 uppercase tracking-widest mt-0.5">confidence</p>
      </div>
    </div>
  );
}

export default function ResultCard({ result }: Props) {
  const isFake  = result.is_fake;
  const color   = isFake ? "#EF4444" : "#22C55E";
  const bgGlow  = isFake ? "rgba(239,68,68,0.08)" : "rgba(34,197,94,0.08)";
  const border  = isFake ? "rgba(239,68,68,0.25)" : "rgba(34,197,94,0.25)";
  const Icon    = isFake ? ShieldAlert : ShieldCheck;

  return (
    <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="w-full rounded-2xl p-6 space-y-6"
      style={{ background: "rgba(255,255,255,0.03)", border: `1px solid ${border}`,
               boxShadow: `0 0 40px ${bgGlow}`, backdropFilter: "blur(12px)" }}>

      <div className="flex flex-col items-center gap-5">
        <div className="relative">
          <motion.div animate={{ scale: [1,1.15,1], opacity: [0.4,0.15,0.4] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="absolute inset-0 rounded-full" style={{ background: color }} />
          <motion.div initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 260, damping: 20 }}
            className="relative flex items-center gap-2.5 px-6 py-3 rounded-full font-bold text-xl tracking-wide"
            style={{ background: `${color}18`, border: `2px solid ${color}55`, color }}>
            <Icon size={22} />
            {result.prediction}
          </motion.div>
        </div>
        <ConfidenceRing value={result.confidence} isFake={isFake} />
      </div>

      <div className="grid grid-cols-2 gap-3">
        <StatCell label="Raw score"
          value={result.raw_score != null ? result.raw_score.toFixed(4) : "N/A"}
          icon={<Crosshair size={13} className="text-zinc-500" />} />
        <StatCell label="Processing"
          value={`${(result.processing_time * 1000).toFixed(0)} ms`}
          icon={<Clock size={13} className="text-zinc-500" />} />
        <StatCell label="Frames analyzed"
          value={`${result.frames_analyzed} frames`}
          icon={<Film size={13} className="text-zinc-500" />} />
        <StatCell label="Model"
          value="V2 Temporal (AUC 0.998)"
          icon={<ShieldCheck size={13} className="text-zinc-500" />} />
      </div>
    </motion.div>
  );
}

function StatCell({ label, value, icon }: {
  label: string; value: string; icon: React.ReactNode }) {
  return (
    <div className="rounded-xl p-3 bg-white/[0.03] border border-white/[0.06] space-y-1">
      <p className="text-[11px] text-zinc-500 flex items-center gap-1">{icon} {label}</p>
      <p className="text-base font-semibold text-zinc-200">{value}</p>
    </div>
  );
}