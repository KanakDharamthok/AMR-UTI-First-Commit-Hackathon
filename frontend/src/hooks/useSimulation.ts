// Replace or update your simulation handler function
export const handleRunSimulation = () => {
  // 1. Set your Streamlit URL (use local port for testing, or public URL once deployed)
  const streamlitUrl = process.env.NODE_ENV === 'production' 
    ? 'https://your-app.streamlit.app'  // Deployed Streamlit URL
    : 'http://localhost:8501';          // Local Streamlit URL

  // 2. Open Streamlit in a new browser tab
  window.open(streamlitUrl, '_blank');
};