"use client";

// GradCAMDisplay — MVP placeholder
// Will display GradCAM heatmap overlays when the /gradcam endpoint is implemented.

import { motion } from "framer-motion";
import { Layers } from "lucide-react";

export default function GradCAMDisplay() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="w-full rounded-2xl p-8 flex flex-col items-center justify-center gap-4 text-center glass"
      style={{ minHeight: 180 }}
    >
      <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
        <Layers size={28} className="text-indigo-400" />
      </div>
      <div>
        <p className="text-zinc-300 font-medium">GradCAM Visualization</p>
        <p className="text-zinc-600 text-sm mt-1">
          Heatmap overlay support coming in the next update.
        </p>
      </div>
    </motion.div>
  );
}
