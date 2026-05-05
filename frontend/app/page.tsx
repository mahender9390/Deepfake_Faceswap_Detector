"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";
import { Cpu, Loader2, Scan } from "lucide-react";

import VideoUploader   from "@/components/VideoUploader";
import ResultCard      from "@/components/ResultCard";
import LoadingSkeleton from "@/components/LoadingSkeleton";
import { predictVideo, VideoResult } from "@/lib/api";

export default function HomePage() {
  const [videoFile,  setVideoFile]  = useState<File | null>(null);
  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState<VideoResult | null>(null);

  const handleAnalyze = async () => {
    if (!videoFile) return;
    setLoading(true); setResult(null);
    try {
      setResult(await predictVideo(videoFile));
    } catch (err: unknown) {
      toast.error(`Analysis failed: ${err instanceof Error ? err.message : "Unknown error"}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="relative min-h-screen overflow-x-hidden">
      <div className="hero-glow" aria-hidden />
      <div className="pointer-events-none fixed inset-0 opacity-[0.025]"
        style={{
          backgroundImage: "linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
        }} aria-hidden />

      <div className="relative z-10 max-w-2xl mx-auto px-4 py-16 sm:py-24 space-y-10">

        {/* Hero */}
        <section className="text-center space-y-4">

          <motion.h1 initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="text-5xl sm:text-6xl font-extrabold tracking-tight">
            <span className="gradient-text">Deepfake</span>
            <span className="text-zinc-100">Detector</span>
          </motion.h1>

          <motion.p initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="text-zinc-400 text-lg max-w-md mx-auto leading-relaxed">
            Upload a video to instantly detect if the face is{" "}
            <span className="text-real font-medium">real</span> or{" "}
            <span className="text-fake font-medium">AI-generated</span>.
          </motion.p>
        </section>

        {/* Upload card */}
        <motion.section initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="glass rounded-3xl p-6 sm:p-8 space-y-6">

          <VideoUploader
            onFileSelect={(f) => { setVideoFile(f); setResult(null); }}
            disabled={loading}
          />

          <motion.button onClick={handleAnalyze} disabled={!videoFile || loading}
            whileHover={videoFile && !loading ? { scale: 1.02 } : {}}
            whileTap={videoFile && !loading ? { scale: 0.98 } : {}}
            className={`w-full py-3.5 rounded-xl font-semibold text-base flex items-center justify-center gap-2.5 transition-all duration-200
              ${videoFile && !loading
                ? "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20"
                : "bg-white/[0.06] text-zinc-600 cursor-not-allowed"}`}>
            {loading
              ? <><Loader2 size={18} className="animate-spin" />Analysing…</>
              : <><Scan size={18} />Analyse Video</>}
          </motion.button>
        </motion.section>

        {/* Result */}
        <AnimatePresence mode="wait">
          {loading && (
            <motion.div key="skeleton" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <LoadingSkeleton />
            </motion.div>
          )}
          {result && !loading && (
            <motion.div key="result"
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.4 }}>
              <ResultCard result={result} />
            </motion.div>
          )}
        </AnimatePresence>

        <footer className="text-center text-xs text-zinc-700 pb-4">
          Deepfake Detector · EfficientNet-B4 + Frame Difference Transformer · PyTorch &amp; Next.js
        </footer>
      </div>
    </main>
  );
}