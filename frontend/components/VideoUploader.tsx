"use client";

import { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Video, X, CheckCircle2, Film } from "lucide-react";

interface Props {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

const MAX_BYTES  = 100 * 1024 * 1024; // 100 MB
const ACCEPT     = ["video/mp4", "video/quicktime", "video/x-msvideo"];
const ACCEPT_EXT = ".mp4,.mov,.avi";

function formatBytes(bytes: number) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function VideoUploader({ onFileSelect, disabled }: Props) {
  const [dragActive, setDragActive] = useState(false);
  const [file,       setFile]       = useState<File | null>(null);
  const [duration,   setDuration]   = useState<string | null>(null);
  const [error,      setError]      = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const processFile = useCallback(
    (f: File) => {
      setError(null);

      if (!ACCEPT.includes(f.type) && !ACCEPT_EXT.split(",").some(ext => f.name.toLowerCase().endsWith(ext.replace(".", "")))) {
        setError("Only MP4, MOV, and AVI videos are supported.");
        return;
      }
      if (f.size > MAX_BYTES) {
        setError(`File too large. Max 100 MB (yours: ${formatBytes(f.size)}).`);
        return;
      }

      setFile(f);
      setDuration(null);
      onFileSelect(f);

      // Probe duration via a hidden video element
      const url = URL.createObjectURL(f);
      const vid = document.createElement("video");
      vid.preload = "metadata";
      vid.onloadedmetadata = () => {
        const d = vid.duration;
        if (isFinite(d)) {
          const m = Math.floor(d / 60);
          const s = Math.floor(d % 60);
          setDuration(`${m}:${s.toString().padStart(2, "0")}`);
        }
        URL.revokeObjectURL(url);
      };
      vid.src = url;
    },
    [onFileSelect]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const f = e.dataTransfer.files[0];
      if (f) processFile(f);
    },
    [processFile]
  );

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    setDuration(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="w-full">
      <motion.div
        onClick={() => !disabled && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={onDrop}
        animate={{
          borderColor: dragActive
            ? "rgba(99,102,241,0.9)"
            : error
            ? "rgba(239,68,68,0.5)"
            : file
            ? "rgba(34,197,94,0.5)"
            : "rgba(255,255,255,0.08)",
          backgroundColor: dragActive ? "rgba(99,102,241,0.07)" : "rgba(255,255,255,0.02)",
        }}
        transition={{ duration: 0.2 }}
        className={`relative w-full rounded-2xl border-2 border-dashed cursor-pointer
          overflow-hidden select-none
          ${disabled ? "opacity-50 pointer-events-none" : "hover:border-indigo-500/50 hover:bg-white/[0.03]"}`}
        style={{ minHeight: "220px" }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_EXT}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) processFile(f); }}
          className="hidden"
          disabled={disabled}
        />

        <AnimatePresence mode="wait">
          {file ? (
            /* ── file selected state ────────────────── */
            <motion.div
              key="selected"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1,  scale: 1 }}
              exit={{    opacity: 0,  scale: 0.96 }}
              className="flex flex-col items-center justify-center gap-5 p-10"
            >
              <button
                onClick={clearFile}
                className="absolute top-3 right-3 p-1.5 rounded-full
                  bg-zinc-900/80 border border-white/10 hover:bg-zinc-800
                  text-zinc-400 hover:text-white transition-colors"
              >
                <X size={14} />
              </button>

              {/* Icon */}
              <div className="p-4 rounded-2xl bg-real/10 border border-real/20">
                <Film size={32} className="text-real" />
              </div>

              {/* File info */}
              <div className="text-center space-y-1">
                <div className="flex items-center gap-2 justify-center">
                  <CheckCircle2 size={16} className="text-real shrink-0" />
                  <p className="text-zinc-200 font-medium truncate max-w-[260px]">
                    {file.name}
                  </p>
                </div>
                <div className="flex items-center gap-3 justify-center text-sm text-zinc-500">
                  <span>{formatBytes(file.size)}</span>
                  {duration && (
                    <>
                      <span className="text-zinc-700">·</span>
                      <span>{duration}</span>
                    </>
                  )}
                </div>
              </div>
            </motion.div>
          ) : (
            /* ── idle state ─────────────────────────── */
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{    opacity: 0 }}
              className="flex flex-col items-center justify-center gap-4 p-10"
            >
              <motion.div
                animate={{ y: dragActive ? -8 : 0 }}
                transition={{ type: "spring", stiffness: 300 }}
                className="p-4 rounded-2xl bg-indigo-500/10 border border-indigo-500/20"
              >
                {dragActive ? (
                  <Upload size={32} className="text-indigo-400" />
                ) : (
                  <Video size={32} className="text-indigo-400" />
                )}
              </motion.div>

              <div className="text-center">
                <p className="text-zinc-200 font-medium">
                  {dragActive ? "Drop it here!" : "Drop a video or click to browse"}
                </p>
                <p className="text-zinc-500 text-sm mt-1">MP4, MOV, AVI · Max 100 MB</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{    opacity: 0, y: -4 }}
            className="mt-2 text-sm text-red-400 flex items-center gap-1.5 px-1"
          >
            <X size={13} /> {error}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
}
