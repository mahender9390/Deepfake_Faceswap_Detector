/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow images from localhost (for future GradCAM integration)
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost" },
    ],
  },
};

module.exports = nextConfig;
