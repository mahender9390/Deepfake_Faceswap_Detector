"use client";

import { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, Image as ImageIcon, X, CheckCircle2 } from "lucide-react";

interface Props {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

const MAX_BYTES = 10 * 1024 * 1024; // 10 MB
const ACCEPT    = ["image/jpeg", "image/jpg", "image/png"];
const ACCEPT_EXT = ".jpg,.jpeg,.png";

function formatBytes(bytes: number) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function ImageUploader({ onFileSelect, disabled }: Props) {
  const [dragActive, setDragActive] = useState(false);
  const [preview,    setPreview]    = useState<string | null>(null);
  const [file,       setFile]       = useState<File | null>(null);
  const [error,      setError]      = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    (f: File) => {
      setError(null);
      if (!ACCEPT.includes(f.type)) {
        setError("Only JPG and PNG images are supported.");
        return;
      }
      if (f.size > MAX_BYTES) {
        setError(`File too large. Max 10 MB (yours: ${formatBytes(f.size)}).`);
        return;
      }
      setFile(f);
      const url = URL.createObjectURL(f);
      setPreview(url);
      onFileSelect(f);
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

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) processFile(f);
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    setPreview(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="w-full">
      {/* Drop zone */}
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
            : preview
            ? "rgba(34,197,94,0.5)"
            : "rgba(255,255,255,0.08)",
          backgroundColor: dragActive
            ? "rgba(99,102,241,0.07)"
            : "rgba(255,255,255,0.02)",
        }}
        transition={{ duration: 0.2 }}
        className={`relative w-full rounded-2xl border-2 border-dashed cursor-pointer
          overflow-hidden transition-all select-none
          ${disabled ? "opacity-50 pointer-events-none" : "hover:border-indigo-500/50 hover:bg-white/[0.03]"}`}
        style={{ minHeight: preview ? "auto" : "220px" }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_EXT}
          onChange={onInputChange}
          className="hidden"
          disabled={disabled}
        />

        <AnimatePresence mode="wait">
          {preview ? (
            /* ── preview state ──────────────────────── */
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1,  scale: 1 }}
              exit={{    opacity: 0,  scale: 0.96 }}
              className="relative p-4"
            >
              {/* Clear button */}
              <button
                onClick={clearFile}
                className="absolute top-3 right-3 z-10 p-1.5 rounded-full
                  bg-zinc-900/80 border border-white/10 hover:bg-zinc-800
                  text-zinc-400 hover:text-white transition-colors"
              >
                <X size={14} />
              </button>

              <img
                src={preview}
                alt="Preview"
                className="w-full max-h-72 object-contain rounded-xl"
              />

              {/* File meta */}
              <div className="mt-3 flex items-center gap-2 text-sm">
                <CheckCircle2 size={16} className="text-real shrink-0" />
                <span className="text-zinc-300 truncate">{file?.name}</span>
                <span className="text-zinc-500 shrink-0">{formatBytes(file?.size ?? 0)}</span>
              </div>
            </motion.div>
          ) : (
            /* ── idle / drag state ──────────────────── */
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
                  <ImageIcon size={32} className="text-indigo-400" />
                )}
              </motion.div>

              <div className="text-center">
                <p className="text-zinc-200 font-medium">
                  {dragActive ? "Drop it here!" : "Drop an image or click to browse"}
                </p>
                <p className="text-zinc-500 text-sm mt-1">JPG, PNG · Max 10 MB</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Error message */}
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
