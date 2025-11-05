/********
 Tailwind CSS Config
********/
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx,js,jsx}"
  ],
  theme: {
    extend: {
      colors: {
        fake: "#e74c3c",
        real: "#2ecc71",
        suspect: "#f39c12"
      }
    }
  },
  plugins: []
}
