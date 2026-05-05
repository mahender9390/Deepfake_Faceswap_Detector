import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Toaster } from "sonner";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Deepfake Detector — AI Face Authenticity Analysis",
  description:
    "Upload an image or video to instantly detect whether a face is real or AI-generated using our state-of-the-art EfficientNet + Transformer model.",
  keywords: ["deepfake", "detection", "AI", "face authenticity", "fake image detector"],
  openGraph: {
    title: "Deepfake Detector",
    description: "AI-powered face authenticity detection",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="bg-[#0a0a0a] text-zinc-100 antialiased">
        {children}

        {/* Toast notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            style: {
              background: "#18181b",
              border: "1px solid rgba(255,255,255,0.08)",
              color: "#f4f4f5",
            },
          }}
        />
      </body>
    </html>
  );
}
